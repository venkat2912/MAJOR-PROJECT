#include "model.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace waste {

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[tensorrt] %s\n", msg);
    }
};

class TrtModel : public Model {
public:
    TrtModel(const std::string& path, int batch, int size)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open engine file " + path);
        const std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("cannot create the TensorRT runtime");
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_)
            throw std::runtime_error("cannot load " + path +
                                     "; engines only load on the TensorRT version and GPU "
                                     "model they were built with");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("cannot create a TensorRT execution context");

        nvinfer1::Dims in{}, out{};
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error(std::string("tensor ") + name + " is not float32");
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                in = engine_->getTensorShape(name);
            } else {
                output_name_ = name;
                out = engine_->getTensorShape(name);
            }
        }
        if (engine_->getNbIOTensors() != 2 || input_name_.empty() || output_name_.empty())
            throw std::runtime_error("expected an engine with one input and one output");
        if (in.nbDims != 4 || in.d[0] != batch || in.d[1] != 3 || in.d[2] != size || in.d[3] != size)
            throw std::runtime_error("engine input shape does not match --batch and --size");
        if (out.nbDims != 3 || out.d[0] != batch || out.d[1] <= 4 || out.d[2] <= 0)
            throw std::runtime_error(
                "unexpected engine output shape; expected [batch, 4 + classes, anchors]");
        num_classes_ = (int)out.d[1] - 4;
        num_anchors_ = (int)out.d[2];

        const size_t bytes = (size_t)out.d[0] * out.d[1] * out.d[2] * sizeof(float);
        if (cudaMalloc(&d_output_, bytes) != cudaSuccess)
            throw std::runtime_error("cannot allocate the engine output buffer");
        if (!context_->setTensorAddress(output_name_.c_str(), d_output_))
            throw std::runtime_error("cannot bind the engine output");
    }

    ~TrtModel() override { cudaFree(d_output_); }

    const float* infer(float* d_input) override
    {
        if (!context_->setTensorAddress(input_name_.c_str(), d_input))
            throw std::runtime_error("cannot bind the engine input");
        // Default stream, the same one the kernels run on.
        if (!context_->enqueueV3(nullptr)) throw std::runtime_error("TensorRT inference failed");
        return d_output_;
    }

    int num_classes() const override { return num_classes_; }
    int num_anchors() const override { return num_anchors_; }

private:
    // Declared in the order they must outlive each other.
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_, output_name_;
    int num_classes_ = 0, num_anchors_ = 0;
    float* d_output_ = nullptr;
};

}  // namespace

std::unique_ptr<Model> load_tensorrt(const std::string& path, int batch, int size)
{
    return std::make_unique<TrtModel>(path, batch, size);
}

}  // namespace waste
