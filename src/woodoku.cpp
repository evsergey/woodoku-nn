#include "solver.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

auto tik()
{
    static auto t = std::chrono::high_resolution_clock::now();
    const auto old_t = t;
    t = std::chrono::high_resolution_clock::now();
    return old_t;
}

auto& tok(std::ostream& str)
{
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tik()).count();
    return str << dt << " us";
}

int main()
{
    Field field;
    std::ifstream ifs("figures.txt");
    const auto figures = read_figures(ifs);
    std::cout << figures.size() << " figures loaded" << std::endl;
    auto factory = create_solver_factory("model.torch");
    std::shared_ptr<basic_solver> solver;
    if (factory->get_cuda_count() > 0)
    {
        std::cout << factory->get_cuda_count() << " CUDA device(s) found" << std::endl;
        solver = factory->get_cuda_solver(0);
    }
    else
    {
        std::cout << "CUDA is not avalable" << std::endl;
        solver = factory->get_cpu_solver();
    }
    std::cout << "Scripted model loaded" << std::endl;
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<size_t> distr(0, figures.size() - 1);
    std::vector<Choice> choices;
    size_t total_score = 0;
    size_t move;
    for (move = 1; ; ++move)
    {
        const TriFigures figs = {
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)])
        };
        if (!std::any_of(figs.begin(), figs.end(), [&field](const Figure& fig) { return field.has_placements(fig); }))
            continue;
        std::cout << "#" << move << " ========================="
            << std::endl << figs;
        size_t score;
        tik();
        auto fields = field.get_all_next(figs, score, &choices);
        std::cout << "Alternatives: " << fields.size() << ". Search: " << tok;
        total_score += score;
        if (fields.empty())
            break;
        Field::random_shrink(fields, 10000, rng);
        tik();
        auto best = solver->choose_best(fields);
        std::cout << ". Evaluation: " << tok << std::endl;
        const auto& choice = choices[best];
        field.print_choice(std::cout, figs, choice);
        field = fields[best];
        std::cout << "Score: " << total_score << std::endl << std::endl;
    }
    std::cout << std::endl << "Total moves: " << move << ". Total score: " << total_score << std::endl;
}