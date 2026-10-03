#include "gpu_pipeline.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <torch/cuda.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using waste::Detection;
using waste::GpuPipeline;
using waste::Tile;

struct Args {
    std::string model, classes, input, out, json, csv;
    int batch = 8, size = 640, repeat = 1;
    float overlap = 0.2f, conf = 0.25f, iou = 0.5f;
    bool ios = false, agnostic = false, full = true;
};

struct Timings {
    double upload = 0, preprocess = 0, infer = 0, decode = 0, nms = 0, coverage = 0;
    double total() const { return upload + preprocess + infer + decode + nms + coverage; }
    void scale(double f)
    {
        upload *= f; preprocess *= f; infer *= f; decode *= f; nms *= f; coverage *= f;
    }
};

struct FrameResult {
    std::vector<Detection> dets;
    float coverage = 0.0f;
    int tiles = 0;
};

// GPU work is asynchronous, so every lap waits for the device first.
class StageClock {
public:
    explicit StageClock(GpuPipeline& pipe) : pipe_(pipe), last_(clock::now()) {}
    double lap()
    {
        pipe_.sync();
        const auto now = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;
        return ms;
    }

private:
    using clock = std::chrono::steady_clock;
    GpuPipeline& pipe_;
    clock::time_point last_;
};

static void usage()
{
    std::puts(
        "waste_engine --model M.torchscript --input IMAGE_OR_VIDEO [options]\n"
        "  --classes FILE   class names, one per line\n"
        "  --batch N        batch size the model was exported with (default 8)\n"
        "  --size N         network input size (default 640)\n"
        "  --overlap F      tile overlap fraction (default 0.2)\n"
        "  --conf F         confidence threshold (default 0.25)\n"
        "  --iou F          overlap threshold for suppression (default 0.5)\n"
        "  --ios            match on intersection over smaller box instead of IoU\n"
        "  --agnostic       suppress across classes\n"
        "  --no-full        skip the extra whole-frame tile\n"
        "  --repeat N       run an image N times and report mean timings\n"
        "  --out FILE       annotated image (image input only)\n"
        "  --json FILE      detections and timings (image input only)\n"
        "  --csv FILE       per-frame scores (video input only)");
}

static Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--model") a.model = next();
        else if (k == "--classes") a.classes = next();
        else if (k == "--input") a.input = next();
        else if (k == "--out") a.out = next();
        else if (k == "--json") a.json = next();
        else if (k == "--csv") a.csv = next();
        else if (k == "--batch") a.batch = std::stoi(next());
        else if (k == "--size") a.size = std::stoi(next());
        else if (k == "--repeat") a.repeat = std::stoi(next());
        else if (k == "--overlap") a.overlap = std::stof(next());
        else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--iou") a.iou = std::stof(next());
        else if (k == "--ios") a.ios = true;
        else if (k == "--agnostic") a.agnostic = true;
        else if (k == "--no-full") a.full = false;
        else throw std::runtime_error("unknown option " + k);
    }
    if (a.model.empty() || a.input.empty()) throw std::runtime_error("--model and --input are required");
    return a;
}

static std::vector<int> tile_starts(int length, int size, int step)
{
    std::vector<int> starts;
    if (length <= size) {
        starts.push_back(0);
        return starts;
    }
    for (int p = 0; p + size < length; p += step) starts.push_back(p);
    starts.push_back(length - size);  // last tile sits flush with the edge
    return starts;
}

// Overlapping full-resolution tiles, plus one downscaled whole-frame tile so
// objects larger than a tile are still seen. Mirrored in python/baseline.py.
static std::vector<Tile> make_tiles(int W, int H, int S, float overlap, bool add_full)
{
    const int step = std::max(1, (int)(S * (1.0f - overlap)));
    std::vector<Tile> tiles;
    for (int y : tile_starts(H, S, step))
        for (int x : tile_starts(W, S, step))
            tiles.push_back({(float)x, (float)y, 1.0f, 0.0f, 0.0f,
                             (float)std::min(S, W), (float)std::min(S, H)});
    if (add_full && (W > S || H > S)) {
        const float scale = (float)std::max(W, H) / (float)S;
        const float cw = W / scale, ch = H / scale;
        tiles.push_back({0.0f, 0.0f, scale, std::floor((S - cw) / 2.0f),
                         std::floor((S - ch) / 2.0f), cw, ch});
    }
    return tiles;
}

static FrameResult process(const cv::Mat& frame, const Args& a, torch::jit::Module& module,
                           GpuPipeline& pipe, Timings& t)
{
    const cv::Mat bgr = frame.isContinuous() ? frame : frame.clone();
    FrameResult r;
    StageClock clk(pipe);

    pipe.upload_frame(bgr.data, bgr.cols, bgr.rows);
    const std::vector<Tile> tiles = make_tiles(bgr.cols, bgr.rows, a.size, a.overlap, a.full);
    pipe.set_tiles(tiles.data(), (int)tiles.size());
    r.tiles = (int)tiles.size();
    t.upload += clk.lap();

    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    for (int first = 0; first < r.tiles; first += a.batch) {
        const int count = std::min(a.batch, r.tiles - first);

        float* d_input = pipe.preprocess(first, count);
        t.preprocess += clk.lap();

        // Wraps the device buffer without copying it.
        torch::Tensor input = torch::from_blob(d_input, {a.batch, 3, a.size, a.size}, opts);
        torch::jit::IValue out = module.forward({input});
        torch::Tensor pred = out.isTuple() ? out.toTuple()->elements()[0].toTensor() : out.toTensor();
        pred = pred.to(torch::kFloat32).contiguous();
        if (pred.dim() != 3 || pred.size(0) != a.batch || pred.size(1) <= 4)
            throw std::runtime_error(
                "unexpected model output shape; expected [batch, 4 + classes, anchors] "
                "from a YOLO detect model exported with the same --batch");
        t.infer += clk.lap();

        pipe.decode(pred.data_ptr<float>(), (int)pred.size(1) - 4, (int)pred.size(2),
                    first, count, a.conf);
        t.decode += clk.lap();
    }

    r.dets = pipe.nms(a.iou, a.ios, a.agnostic);
    t.nms += clk.lap();
    r.coverage = pipe.coverage(r.dets);
    t.coverage += clk.lap();
    return r;
}

static std::vector<std::string> load_classes(const std::string& path)
{
    std::vector<std::string> names;
    if (path.empty()) return names;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open classes file " + path);
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

static std::string class_name(const std::vector<std::string>& names, int cls)
{
    return cls >= 0 && cls < (int)names.size() ? names[cls] : std::to_string(cls);
}

static void print_timings(const Timings& t, int frames)
{
    std::printf("mean over %d run(s), ms per frame\n", frames);
    std::printf("  upload      %8.3f\n", t.upload);
    std::printf("  preprocess  %8.3f\n", t.preprocess);
    std::printf("  infer       %8.3f\n", t.infer);
    std::printf("  decode      %8.3f\n", t.decode);
    std::printf("  nms         %8.3f\n", t.nms);
    std::printf("  coverage    %8.3f\n", t.coverage);
    std::printf("  total       %8.3f  (%.1f fps)\n", t.total(), 1000.0 / t.total());
}

static void write_json(const std::string& path, const Args& a, const cv::Mat& img,
                       const FrameResult& r, const Timings& t,
                       const std::vector<std::string>& names)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << "{\n  \"image\": \"" << a.input << "\",\n  \"width\": " << img.cols
      << ",\n  \"height\": " << img.rows << ",\n  \"tiles\": " << r.tiles
      << ",\n  \"coverage_pct\": " << r.coverage * 100.0f << ",\n  \"timing_ms\": {"
      << "\"upload\": " << t.upload << ", \"preprocess\": " << t.preprocess
      << ", \"infer\": " << t.infer << ", \"decode\": " << t.decode
      << ", \"nms\": " << t.nms << ", \"coverage\": " << t.coverage
      << "},\n  \"detections\": [\n";
    for (size_t i = 0; i < r.dets.size(); ++i) {
        const Detection& d = r.dets[i];
        f << "    {\"cls\": " << d.cls << ", \"name\": \"" << class_name(names, d.cls)
          << "\", \"score\": " << d.score << ", \"box\": [" << d.x1 << ", " << d.y1
          << ", " << d.x2 << ", " << d.y2 << "]}" << (i + 1 < r.dets.size() ? "," : "")
          << "\n";
    }
    f << "  ]\n}\n";
}

static void write_annotated(const std::string& path, const cv::Mat& img, const FrameResult& r,
                            const std::vector<std::string>& names)
{
    cv::Mat vis = img.clone();
    const int thick = std::max(2, (int)std::lround(std::max(img.cols, img.rows) / 600.0));
    for (const Detection& d : r.dets) {
        const cv::Scalar color((d.cls * 67 + 80) % 256, (d.cls * 131 + 160) % 256,
                               (d.cls * 197 + 40) % 256);
        cv::rectangle(vis, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2),
                      color, thick);
        char label[128];
        std::snprintf(label, sizeof(label), "%s %.2f", class_name(names, d.cls).c_str(), d.score);
        cv::putText(vis, label, cv::Point((int)d.x1, std::max(12, (int)d.y1 - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4 * thick, color, thick);
    }
    if (!cv::imwrite(path, vis)) throw std::runtime_error("cannot write " + path);
}

static bool is_video(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv";
}

static int run_image(const Args& a, torch::jit::Module& module, GpuPipeline& pipe,
                     const std::vector<std::string>& names)
{
    const cv::Mat img = cv::imread(a.input, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read image " + a.input);

    // The first runs include CUDA start-up and TorchScript optimisation.
    const int warmup = a.repeat > 3 ? 2 : 0;
    Timings t;
    FrameResult r;
    for (int i = 0; i < a.repeat; ++i) {
        Timings one;
        r = process(img, a, module, pipe, one);
        if (i >= warmup) {
            t.upload += one.upload; t.preprocess += one.preprocess; t.infer += one.infer;
            t.decode += one.decode; t.nms += one.nms; t.coverage += one.coverage;
        }
    }
    const int measured = a.repeat - warmup;
    t.scale(1.0 / measured);

    std::vector<int> per_class;
    for (const Detection& d : r.dets) {
        if (d.cls >= (int)per_class.size()) per_class.resize(d.cls + 1, 0);
        ++per_class[d.cls];
    }
    std::printf("%s: %dx%d, %d tiles, %zu objects, %.2f%% of the frame covered\n",
                a.input.c_str(), img.cols, img.rows, r.tiles, r.dets.size(),
                r.coverage * 100.0f);
    for (size_t c = 0; c < per_class.size(); ++c)
        if (per_class[c]) std::printf("  %-20s %d\n", class_name(names, (int)c).c_str(), per_class[c]);
    print_timings(t, measured);

    if (!a.json.empty()) write_json(a.json, a, img, r, t, names);
    if (!a.out.empty()) write_annotated(a.out, img, r, names);
    return 0;
}

static int run_video(const Args& a, torch::jit::Module& module, GpuPipeline& pipe)
{
    cv::VideoCapture cap(a.input);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video " + a.input);

    std::ofstream csv;
    if (!a.csv.empty()) {
        csv.open(a.csv);
        if (!csv) throw std::runtime_error("cannot write " + a.csv);
        csv << "frame,objects,coverage_pct\n";
    }

    const int warmup = 2;
    Timings t;
    int frames = 0, measured = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        Timings one;
        const FrameResult r = process(frame, a, module, pipe, one);
        if (csv.is_open()) csv << frames << "," << r.dets.size() << "," << r.coverage * 100.0f << "\n";
        if (frames >= warmup) {
            t.upload += one.upload; t.preprocess += one.preprocess; t.infer += one.infer;
            t.decode += one.decode; t.nms += one.nms; t.coverage += one.coverage;
            ++measured;
        }
        ++frames;
    }
    std::printf("%s: %d frames\n", a.input.c_str(), frames);
    if (measured > 0) {
        t.scale(1.0 / measured);
        print_timings(t, measured);
    }
    return 0;
}

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        const Args a = parse_args(argc, argv);
        if (!torch::cuda::is_available()) throw std::runtime_error("no CUDA device available");

        torch::jit::Module module = torch::jit::load(a.model, torch::kCUDA);
        module.eval();
        torch::NoGradGuard no_grad;

        GpuPipeline pipe(a.size, a.batch);
        const std::vector<std::string> names = load_classes(a.classes);
        return is_video(a.input) ? run_video(a, module, pipe) : run_image(a, module, pipe, names);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
