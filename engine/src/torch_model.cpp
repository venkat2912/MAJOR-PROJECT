#include "model.h"

#include <torch/cuda.h>
#include <torch/script.h>

#include <stdexcept>

namespace waste {

namespace {

class TorchModel : public Model {
public:
    TorchModel(const std::string& path, int batch, int size) : batch_(batch), size_(size)
    {
        if (!torch::cuda::is_available()) throw std::runtime_error("no CUDA device available");
        module_ = torch::jit::load(path, torch::kCUDA);
        module_.eval();
    }

    const float* infer(float* d_input) override
    {
        torch::NoGradGuard no_grad;
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
        // Wraps the device buffer without copying it.
        torch::Tensor input = torch::from_blob(d_input, {batch_, 3, size_, size_}, opts);
        torch::jit::IValue out = module_.forward({input});
        pred_ = out.isTuple() ? out.toTuple()->elements()[0].toTensor() : out.toTensor();
        pred_ = pred_.to(torch::kFloat32).contiguous();
        if (pred_.dim() != 3 || pred_.size(0) != batch_ || pred_.size(1) <= 4)
            throw std::runtime_error(
                "unexpected model output shape; expected [batch, 4 + classes, anchors] "
                "from a YOLO detect model exported with the same --batch");
        return pred_.data_ptr<float>();
    }

    int num_classes() const override { return (int)pred_.size(1) - 4; }
    int num_anchors() const override { return (int)pred_.size(2); }

private:
    int batch_, size_;
    torch::jit::Module module_;
    torch::Tensor pred_;  // keeps the output alive while the kernels read it
};

}  // namespace

std::unique_ptr<Model> load_torchscript(const std::string& path, int batch, int size)
{
    return std::make_unique<TorchModel>(path, batch, size);
}

}  // namespace waste
