#include "gpu_pipeline.h"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err_) + " at " +      \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

namespace waste {

namespace {

const int kCoverageStride = 4;  // coverage is sampled on a 1/4 resolution grid

// One thread per output pixel, one grid layer (blockIdx.z) per batch slot.
__global__ void preprocess_kernel(const uint8_t* frame, int fw, int fh,
                                  const Tile* tiles, int n_valid, int S,
                                  float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z;
    if (x >= S || y >= S) return;

    const float pad = 114.0f / 255.0f;
    float r = pad, g = pad, b = pad;

    if (t < n_valid) {
        const Tile tl = tiles[t];
        const float cx = x - tl.pad_x;
        const float cy = y - tl.pad_y;
        if (cx >= 0.0f && cy >= 0.0f && cx < tl.content_w && cy < tl.content_h) {
            // Half-pixel centres, so scale == 1 is an exact copy.
            float sx = tl.src_x + (cx + 0.5f) * tl.scale - 0.5f;
            float sy = tl.src_y + (cy + 0.5f) * tl.scale - 0.5f;
            sx = fminf(fmaxf(sx, 0.0f), fw - 1.0f);
            sy = fminf(fmaxf(sy, 0.0f), fh - 1.0f);
            const int x0 = (int)sx, y0 = (int)sy;
            const int x1 = min(x0 + 1, fw - 1), y1 = min(y0 + 1, fh - 1);
            const float ax = sx - x0, ay = sy - y0;

            const uint8_t* p00 = frame + ((size_t)y0 * fw + x0) * 3;
            const uint8_t* p01 = frame + ((size_t)y0 * fw + x1) * 3;
            const uint8_t* p10 = frame + ((size_t)y1 * fw + x0) * 3;
            const uint8_t* p11 = frame + ((size_t)y1 * fw + x1) * 3;

            float c[3];
            for (int k = 0; k < 3; ++k) {
                const float top = p00[k] + ax * (p01[k] - p00[k]);
                const float bot = p10[k] + ax * (p11[k] - p10[k]);
                c[k] = (top + ay * (bot - top)) * (1.0f / 255.0f);
            }
            b = c[0]; g = c[1]; r = c[2];
        }
    }

    const size_t plane = (size_t)S * S;
    float* o = out + (size_t)t * 3 * plane + (size_t)y * S + x;
    o[0] = r;
    o[plane] = g;
    o[2 * plane] = b;
}

// One thread per (tile, anchor). pred is [batch, 4 + nc, A]: rows 0-3 are the
// box centre and size in network pixels, the rest are class scores.
__global__ void decode_kernel(const float* pred, int nc, int A,
                              const Tile* tiles, int n_valid, float conf,
                              float fw, float fh, Detection* cand, int max_cand,
                              int* count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_valid * A) return;
    const int b = idx / A, a = idx % A;
    const float* p = pred + (size_t)b * (4 + nc) * A + a;

    float best = p[(size_t)4 * A];
    int cls = 0;
    for (int c = 1; c < nc; ++c) {
        const float s = p[(size_t)(4 + c) * A];
        if (s > best) { best = s; cls = c; }
    }
    if (best < conf) return;

    const Tile tl = tiles[b];
    const float cx = p[0], cy = p[A], w = p[(size_t)2 * A], h = p[(size_t)3 * A];
    float x1 = (cx - 0.5f * w - tl.pad_x) * tl.scale + tl.src_x;
    float y1 = (cy - 0.5f * h - tl.pad_y) * tl.scale + tl.src_y;
    float x2 = (cx + 0.5f * w - tl.pad_x) * tl.scale + tl.src_x;
    float y2 = (cy + 0.5f * h - tl.pad_y) * tl.scale + tl.src_y;
    x1 = fminf(fmaxf(x1, 0.0f), fw);
    y1 = fminf(fmaxf(y1, 0.0f), fh);
    x2 = fminf(fmaxf(x2, 0.0f), fw);
    y2 = fminf(fmaxf(y2, 0.0f), fh);
    if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) return;

    const int slot = atomicAdd(count, 1);
    if (slot >= max_cand) return;
    Detection d;
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    d.score = best;
    d.cls = cls;
    cand[slot] = d;
}

struct ByScoreDesc {
    __host__ __device__ bool operator()(const Detection& a, const Detection& b) const
    {
        if (a.score != b.score) return a.score > b.score;
        if (a.x1 != b.x1) return a.x1 < b.x1;
        return a.y1 < b.y1;
    }
};

__device__ inline float overlap(const Detection& a, const Detection& b, bool use_ios)
{
    const float iw = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1);
    const float ih = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1);
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return use_ios ? inter / fminf(area_a, area_b)
                   : inter / (area_a + area_b - inter);
}

// One thread per (box i, 64-box block). Bit k of mask[i * blocks + blk] is set
// when lower-ranked box blk * 64 + k overlaps box i past the threshold.
__global__ void nms_mask_kernel(const Detection* dets, int n, int blocks,
                                float thr, bool use_ios, bool agnostic,
                                unsigned long long* mask)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * blocks) return;
    const int i = idx / blocks, blk = idx % blocks;
    const Detection di = dets[i];

    unsigned long long bits = 0;
    const int start = blk * 64;
    for (int k = 0; k < 64; ++k) {
        const int j = start + k;
        if (j >= n) break;
        if (j <= i) continue;
        const Detection dj = dets[j];
        if (!agnostic && dj.cls != di.cls) continue;
        if (overlap(di, dj, use_ios) > thr) bits |= 1ULL << k;
    }
    mask[idx] = bits;
}

// One thread per sample point on the coarse grid.
__global__ void coverage_kernel(const Detection* dets, int n, int gw, int gh,
                                int stride, unsigned int* covered)
{
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= gw || gy >= gh) return;
    const float px = (gx + 0.5f) * stride;
    const float py = (gy + 0.5f) * stride;
    for (int i = 0; i < n; ++i) {
        const Detection d = dets[i];
        if (px >= d.x1 && px < d.x2 && py >= d.y1 && py < d.y2) {
            atomicAdd(covered, 1u);
            return;
        }
    }
}

}  // namespace

GpuPipeline::GpuPipeline(int net_size, int batch, int max_tiles,
                         int max_candidates, int nms_limit)
    : net_size_(net_size), batch_(batch), max_tiles_(max_tiles),
      max_cand_(max_candidates), nms_limit_(std::min(nms_limit, max_candidates))
{
    const size_t input_bytes = (size_t)batch_ * 3 * net_size_ * net_size_ * sizeof(float);
    const size_t mask_bytes =
        (size_t)nms_limit_ * ((nms_limit_ + 63) / 64) * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc(&d_tiles_, (size_t)max_tiles_ * sizeof(Tile)));
    CUDA_CHECK(cudaMalloc(&d_input_, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_cand_, (size_t)max_cand_ * sizeof(Detection)));
    CUDA_CHECK(cudaMalloc(&d_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask_, mask_bytes));
    CUDA_CHECK(cudaMalloc(&d_covered_, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_count_, 0, sizeof(int)));
}

GpuPipeline::~GpuPipeline()
{
    cudaFree(d_frame_);
    cudaFree(d_tiles_);
    cudaFree(d_input_);
    cudaFree(d_cand_);
    cudaFree(d_count_);
    cudaFree(d_mask_);
    cudaFree(d_covered_);
}

void GpuPipeline::upload_frame(const uint8_t* bgr, int width, int height)
{
    const size_t bytes = (size_t)width * height * 3;
    if (bytes > frame_capacity_) {
        CUDA_CHECK(cudaFree(d_frame_));
        d_frame_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_frame_, bytes));
        frame_capacity_ = bytes;
    }
    CUDA_CHECK(cudaMemcpy(d_frame_, bgr, bytes, cudaMemcpyHostToDevice));
    frame_w_ = width;
    frame_h_ = height;
}

void GpuPipeline::set_tiles(const Tile* tiles, int count)
{
    if (count > max_tiles_)
        throw std::runtime_error("too many tiles for one frame: " + std::to_string(count));
    CUDA_CHECK(cudaMemcpy(d_tiles_, tiles, (size_t)count * sizeof(Tile),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count_, 0, sizeof(int)));
    num_tiles_ = count;
}

float* GpuPipeline::preprocess(int first_tile, int count)
{
    if (count > batch_ || first_tile + count > num_tiles_)
        throw std::runtime_error("preprocess: tile range out of bounds");
    const dim3 block(16, 16);
    const dim3 grid((net_size_ + 15) / 16, (net_size_ + 15) / 16, batch_);
    preprocess_kernel<<<grid, block>>>(d_frame_, frame_w_, frame_h_,
                                       d_tiles_ + first_tile, count, net_size_,
                                       d_input_);
    CUDA_CHECK(cudaGetLastError());
    return d_input_;
}

void GpuPipeline::decode(const float* d_pred, int num_classes, int num_anchors,
                         int first_tile, int count, float conf)
{
    const int threads = count * num_anchors;
    if (threads <= 0) return;
    decode_kernel<<<(threads + 255) / 256, 256>>>(
        d_pred, num_classes, num_anchors, d_tiles_ + first_tile, count, conf,
        (float)frame_w_, (float)frame_h_, d_cand_, max_cand_, d_count_);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<Detection> GpuPipeline::nms(float threshold, bool use_ios, bool class_agnostic,
                                        bool merge)
{
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_count_, sizeof(int), cudaMemcpyDeviceToHost));
    int n = std::min(count, max_cand_);
    if (n <= 0) return {};

    thrust::sort(thrust::device, d_cand_, d_cand_ + n, ByScoreDesc());
    n = std::min(n, nms_limit_);

    const int blocks = (n + 63) / 64;
    const int threads = n * blocks;
    nms_mask_kernel<<<(threads + 255) / 256, 256>>>(d_cand_, n, blocks, threshold,
                                                    use_ios, class_agnostic, d_mask_);
    CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned long long> mask((size_t)threads);
    std::vector<Detection> sorted((size_t)n);
    CUDA_CHECK(cudaMemcpy(mask.data(), d_mask_, mask.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sorted.data(), d_cand_, sorted.size() * sizeof(Detection),
                          cudaMemcpyDeviceToHost));

    // The scan over the mask is sequential by nature, so it stays on the CPU.
    std::vector<unsigned long long> removed((size_t)blocks, 0ULL);
    std::vector<Detection> keep;
    for (int i = 0; i < n; ++i) {
        if ((removed[i / 64] >> (i % 64)) & 1ULL) continue;
        Detection d = sorted[i];
        const unsigned long long* row = mask.data() + (size_t)i * blocks;
        for (int b = i / 64; b < blocks; ++b) {
            if (merge) {
                // Grow the kept box over every box it absorbs for the first time.
                const unsigned long long fresh = row[b] & ~removed[b];
                for (int k = 0; fresh != 0 && k < 64; ++k) {
                    if (!((fresh >> k) & 1ULL)) continue;
                    const Detection& o = sorted[(size_t)b * 64 + k];
                    d.x1 = std::min(d.x1, o.x1);
                    d.y1 = std::min(d.y1, o.y1);
                    d.x2 = std::max(d.x2, o.x2);
                    d.y2 = std::max(d.y2, o.y2);
                }
            }
            removed[b] |= row[b];
        }
        keep.push_back(d);
    }
    return keep;
}

float GpuPipeline::coverage(const std::vector<Detection>& dets)
{
    const int n = std::min((int)dets.size(), max_cand_);
    if (n == 0 || frame_w_ == 0 || frame_h_ == 0) return 0.0f;

    CUDA_CHECK(cudaMemcpy(d_cand_, dets.data(), (size_t)n * sizeof(Detection),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_covered_, 0, sizeof(unsigned int)));

    const int gw = (frame_w_ + kCoverageStride - 1) / kCoverageStride;
    const int gh = (frame_h_ + kCoverageStride - 1) / kCoverageStride;
    const dim3 block(16, 16);
    const dim3 grid((gw + 15) / 16, (gh + 15) / 16);
    coverage_kernel<<<grid, block>>>(d_cand_, n, gw, gh, kCoverageStride, d_covered_);
    CUDA_CHECK(cudaGetLastError());

    unsigned int covered = 0;
    CUDA_CHECK(cudaMemcpy(&covered, d_covered_, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    return (float)covered / ((float)gw * (float)gh);
}

void GpuPipeline::sync()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace waste
