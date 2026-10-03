#pragma once

#include <cstdint>
#include <vector>

namespace waste {

// One network input cut from the frame. A network pixel (x, y) inside the
// content region maps to the frame pixel
//   (src_x + (x - pad_x) * scale, src_y + (y - pad_y) * scale).
struct Tile {
    float src_x, src_y;          // top-left of the source rect, in frame pixels
    float scale;                 // frame pixels per network pixel
    float pad_x, pad_y;          // letterbox offset inside the network input
    float content_w, content_h;  // valid region size, in network pixels
};

struct Detection {
    float x1, y1, x2, y2;  // frame pixels
    float score;
    int32_t cls;
};

// Owns every device buffer used around the model: the raw frame, the batched
// network input, the detection candidates and the NMS mask. Nothing here
// depends on LibTorch; the model only sees the pointer returned by preprocess().
class GpuPipeline {
public:
    GpuPipeline(int net_size, int batch, int max_tiles = 1024,
                int max_candidates = 16384, int nms_limit = 4096);
    ~GpuPipeline();
    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Copies a continuous BGR8 frame to the device.
    void upload_frame(const uint8_t* bgr, int width, int height);

    // Uploads the tile list for the current frame and clears the candidates.
    void set_tiles(const Tile* tiles, int count);

    // Crop + resize + letterbox + BGR->RGB + /255 + HWC->CHW for `count` tiles
    // starting at `first_tile`, in one kernel. Slots past `count` are filled
    // with padding. Returns a device pointer to [batch, 3, net_size, net_size].
    float* preprocess(int first_tile, int count);

    // Reads raw YOLO output [batch, 4 + num_classes, num_anchors] on the
    // device, thresholds it and appends boxes in frame coordinates.
    void decode(const float* d_pred, int num_classes, int num_anchors,
                int first_tile, int count, float conf);

    // Sorts the candidates by score and suppresses overlaps across all tiles.
    // use_ios switches the overlap metric from IoU to intersection over the
    // smaller box, which merges objects cut by a tile border.
    std::vector<Detection> nms(float threshold, bool use_ios, bool class_agnostic);

    // Fraction of the frame covered by the union of the boxes, in [0, 1].
    float coverage(const std::vector<Detection>& dets);

    // Blocks until all queued GPU work has finished (for timing).
    void sync();

private:
    int net_size_, batch_, max_tiles_, max_cand_, nms_limit_;
    int frame_w_ = 0, frame_h_ = 0;
    size_t frame_capacity_ = 0;
    int num_tiles_ = 0;

    uint8_t* d_frame_ = nullptr;
    Tile* d_tiles_ = nullptr;
    float* d_input_ = nullptr;
    Detection* d_cand_ = nullptr;
    int* d_count_ = nullptr;
    unsigned long long* d_mask_ = nullptr;
    unsigned int* d_covered_ = nullptr;
};

}  // namespace waste
