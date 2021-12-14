#include "solver.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>

namespace
{
    class solver : public basic_solver
    {
        torch::jit::Module& _model;
        torch::Device _device;
        torch::Tensor _tensor;
        std::vector<torch::jit::IValue> _inputs;

    public:
        solver(torch::jit::Module& model, const torch::Device& device)
            : _model(model)
            , _device(device)
            , _inputs(1)
        {
            auto options = torch::
                dtype(torch::kI32)
                .layout(torch::kStrided)
                .requires_grad(false);
            _tensor = torch::empty({ 10000, 3 }, options);
        }

        size_t choose_best(const std::vector<Field>& fields) final
        {
            c10::InferenceMode guard;
            Field::copy_to(fields, _tensor.data<int32_t>(), 0, std::min((size_t)10000, fields.size()));
            using namespace torch::indexing;
            _inputs[0] = _tensor.index({ Slice(None, fields.size()), None, None }).to(_device);
            auto output = _model.forward(_inputs).toTensor().argmax();
            const auto best = static_cast<size_t>(output.item<int>());
            return best;
        }
    };

    class solver_factory_impl : public solver_factory
    {
        std::mutex _mutex;
        const std::string _file_name;
        std::shared_ptr<torch::jit::Module> _loaded_model;
        std::vector<std::shared_ptr<torch::jit::Module>> _models;

    public:
        solver_factory_impl(const std::string& model_file_name)
            : _file_name(model_file_name)
            , _models(1 + (torch::cuda::is_available() ? torch::cuda::device_count() : 0))
        {
        }

        std::shared_ptr<basic_solver> get_cpu_solver() final
        {
            return make_solver(torch::kCPU, 0);
        }

        std::shared_ptr<basic_solver> get_cuda_solver(size_t index) final
        {
            return make_solver(torch::Device(torch::kCUDA, index), index + 1);
        }

        size_t get_cuda_count() const final
        {
            return _models.size() - 1;
        }

    private:
        std::shared_ptr<solver> make_solver(const torch::Device& device, size_t index)
        {
            c10::InferenceMode guard;
            std::lock_guard<std::mutex> lock(_mutex);
            {
                if (!_loaded_model)
                    _models[index] = _loaded_model = std::make_shared<torch::jit::Module>(torch::jit::load(_file_name, device));
                else if (!_models[index])
                    (_models[index] = std::make_shared<torch::jit::Module>(_loaded_model->deepcopy()))->to(device);
            }
            return std::make_shared<solver>(*_models[index], device);
        }
    };
}

std::shared_ptr<solver_factory> create_solver_factory(const std::string& model_file_name)
{
    torch::jit::getExecutorMode() = false;
    return std::make_shared<solver_factory_impl>(model_file_name);
}
