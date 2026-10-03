#pragma once

#include <memory>
#include <string>

namespace waste {

// A detector that reads a device buffer [batch, 3, size, size] and returns a
// device pointer to raw YOLO output [batch, 4 + classes, anchors].
class Model {
public:
    virtual ~Model() = default;

    // The returned pointer stays valid until the next call.
    virtual const float* infer(float* d_input) = 0;

    // Valid after the first infer().
    virtual int num_classes() const = 0;
    virtual int num_anchors() const = 0;
};

std::unique_ptr<Model> load_torchscript(const std::string& path, int batch, int size);

// Only defined when the engine is built with -DWASTE_WITH_TENSORRT=ON.
std::unique_ptr<Model> load_tensorrt(const std::string& path, int batch, int size);

}  // namespace waste
