#include "figure.h"
#include "field.h"

#include <iostream>
#include <fstream>


int main()
{
    using namespace std::literals;
    Field field;
    std::ifstream ifs("figures.txt");
    auto figures = read_figures(ifs);
    //for (auto& f : figures)
    //    std::cout << f << std::endl;

    const Figure& figure = figures[23];
    std::cout << figure;
    field.add(figure, 0, 8);
    field.add(figure, 2, 8);
    field.add(figure, 0, 32);
    field.add(figure, 2, 32);
    std::cout << field << std::endl;

    const auto& a = figures[15];
    const auto& b = figures[25];
    const auto& c = figures[40];
    std::cout << a << "---" << std::endl;
    std::cout << b << "---" << std::endl;
    std::cout << c << "---" << std::endl;

    size_t score;
    auto all_next = field.get_all_next(a, b, c, score);
    for(size_t i=0; i<500; ++i)
        all_next = field.get_all_next(a, b, c, score);
    std::cout << "count=" << all_next.size() << ", score=" << score << std::endl;
    std::cout << *all_next.begin() << std::endl;
}