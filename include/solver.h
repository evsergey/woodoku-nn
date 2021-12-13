#pragma once

#include "figure.h"
#include "field.h"

#include <memory>

struct basic_solver
{
    virtual ~basic_solver() = default;
    virtual size_t choose_best(const std::vector<Field>& fields) = 0;
};

struct solver_factory
{
    virtual ~solver_factory() = default;
    virtual std::shared_ptr<basic_solver> get_cpu_solver() = 0;
    virtual std::shared_ptr<basic_solver> get_cuda_solver(size_t index) = 0;
    virtual size_t get_cuda_count() const = 0;
};

std::shared_ptr<solver_factory> create_solver_factory(const std::string& model_file_name);
