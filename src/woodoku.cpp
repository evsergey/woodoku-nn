#include "figure.h"
#include "field.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

int main()
{
    Field field;
    std::ifstream ifs("figures.txt");
    const auto figures = read_figures(ifs);
    std::cout << figures.size() << " figures loaded" << std::endl;
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
        std::cout << torch::cuda::device_count() << " CUDA device(s) found" << std::endl;
    }
    else
        std::cout << "CUDA is not avalable" << std::endl;
    torch::jit::getBailoutDepth() = 1;
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<size_t> distr(0, figures.size() - 1);
    auto model = torch::jit::load("model.torch", device);
    std::cout << "Scripted model loaded" << std::endl;
    std::vector<Choice> choices;
    size_t total_score = 0;
    auto options = torch::
        dtype(torch::kF32)
        .layout(torch::kStrided)
        .requires_grad(false);
    torch::Tensor tensor = torch::empty({ 10000, 9, 9 }, options);
    size_t move;
    for (move = 1; ; ++move)
    {
        const std::array<std::reference_wrapper<const Figure>, 3> figs = {
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)])
        };
        std::cout << "#" << move << std::endl;
        for (auto& fig : figs)
            std::cout << fig << "---" << std::endl;
        size_t score;
        auto t = std::chrono::high_resolution_clock::now();
        auto fields = field.get_all_next(figs, score, &choices);
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t).count();
        std::cout << "Alternatives: " << fields.size() << ". Time: " << dt << " us" << std::endl;
        total_score += score;
        if (fields.empty())
            break;
        Field::random_shrink(fields, 10000, rng);
        Field::copy_to<float>(fields, reinterpret_cast<float*>(tensor.data_ptr()));
        using namespace torch::indexing;
        auto input = tensor.index({ Slice(None, fields.size()), Slice(), Slice() }).to(device);
        std::vector<torch::jit::IValue> inputs{ input };
        t = std::chrono::high_resolution_clock::now();
        auto output = model.forward(inputs).toTensor();
        dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t).count();
        const auto best = static_cast<size_t>(output[1].item<float>());
        const auto& choice = choices[best];
        field.print_choice(std::cout, figs, choice);
        field = fields[best];
        std::cout << "Score: " << total_score << ". Inference time: " << dt
            << " us\n=======================" << std::endl;
    }
    std::cout << "Total moves: " << move << ". Total score: " << total_score << std::endl;
}