#include "solver.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>

std::mutex global_mutex;
std::vector<Field> all_fields;
std::vector<int32_t> all_scores;
std::atomic_size_t games_count = 0;
size_t scores_sum = 0;
size_t total_games;
size_t ready_games = 0;
std::vector<Figure> figures;
std::chrono::high_resolution_clock::time_point start_time;

int32_t play_game(const Field& start_field, std::shared_ptr<basic_solver> solver, std::vector<Field>& fields, std::vector<int32_t>& scores, std::default_random_engine& rng)
{
    std::uniform_int_distribution<size_t> distr(0, figures.size() - 1);
    auto field = start_field;
    fields.clear();
    scores.clear();
    int32_t total_score = 0;
    while (true)
    {
        const TriFigures figs = {
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)]),
            std::cref(figures[distr(rng)])
        };
        if (!std::any_of(figs.begin(), figs.end(), [&field](const Figure& fig) { return field.has_placements(fig); }))
            continue;
        fields.push_back(field);
        scores.push_back(total_score);
//        std::cout << fields.size() << " ";
        size_t score;
        auto alternatives = field.get_all_next(figs, score);
        total_score += (int32_t)score;
        if (alternatives.empty())
            break;
        Field::random_shrink(alternatives, 10000, rng);
        auto best = solver->choose_best(alternatives);
        field = alternatives[best];
    }
//    std::cout << std::endl;
    for (auto& score : scores)
        score = total_score - score;
    return total_score;
}

void worker(std::shared_ptr<basic_solver> solver, size_t jnum)
{
    std::vector<Field> fields;
    std::vector<int32_t> scores;
    std::random_device rd;
    std::default_random_engine rng(rd());
    while (++games_count <= total_games)
    {
        size_t total_score = play_game(Field(), solver, fields, scores, rng);
        std::lock_guard lock(global_mutex);
        scores_sum += total_score;
        all_fields.insert(all_fields.end(), fields.begin(), fields.end());
        all_scores.insert(all_scores.end(), scores.begin(), scores.end());
        auto t = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t - start_time).count();
        ++ready_games;
        std::cout << "[" << jnum+1 << "] " << ready_games << ": Score=" << total_score << ". T=" << dt << std::endl;
    }
}

int main(int argc, char* argv[])
{
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("figures,f", po::value<std::string>()->default_value("figures.txt"), "path to figures list")
        ("model,m", po::value<std::string>()->default_value("model.torch"), "path to pytorch module")
        ("games,g", po::value<size_t>()->default_value(1000), "number of games to play")
        ("out,o", po::value<std::string>()->default_value("games.data"), "output file")
        ("jobs,j", po::value<size_t>()->default_value(1), "number of working jobs")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    {
        const auto figures_file = vm["figures"].as<std::string>();
        std::ifstream ifs(figures_file);
        if (!ifs)
        {
            std::cerr << "Cannot read figures file: " << figures_file << std::endl;
            return 2;
        }
        figures = read_figures(ifs);
    }
    const auto model_file = vm["model"].as<std::string>();
    auto factory = create_solver_factory(model_file);
    const auto jobs = vm["jobs"].as<size_t>();
    total_games = vm["games"].as<size_t>();
    std::vector<std::jthread> workers;
    workers.reserve(jobs);
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < jobs; ++i)
    {
        auto solver = factory->get_cuda_count() > 0
            ? factory->get_cuda_solver(i % factory->get_cuda_count())
            : factory->get_cpu_solver();
        workers.emplace_back(worker, solver, i);
    }

    for (auto& w : workers)
        w.join();

    std::cout << "Average score: " << scores_sum / total_games << std::endl;
    auto t = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t - start_time).count();
    std::cout << "Average time: " << dt / total_games << " ms" << std::endl;

    const auto output_file = vm["out"].as<std::string>();
    FILE* f = fopen(output_file.c_str(), "wb");
    if (!f)
    {
        std::cerr << "Cannot write output file: " << output_file << std::endl;
        return 3;
    }

    std::vector<int32_t> fields_buffer(all_fields.size() * 3);
    Field::copy_to(all_fields, fields_buffer.data());
    fwrite(fields_buffer.data(), sizeof(fields_buffer[0]), fields_buffer.size(), f);
    fwrite(all_scores.data(), sizeof(all_scores[0]), all_scores.size(), f);
    fclose(f);

    return 0;
}
