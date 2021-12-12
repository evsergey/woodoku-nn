#pragma once
#include <array>
#include <functional>
#include <ostream>
#include <istream>
#include <unordered_set>
#include <vector>

struct Figure
{
    std::array<int32_t, 5> rows;
    size_t weight;
    size_t ncols, nrows;
};

std::ostream& operator<< (std::ostream& str, const Figure& figure);
std::ostream& operator<< (std::ostream& str, const std::array<std::reference_wrapper<const Figure>, 3>& tri_fig);

template<> struct std::hash<Figure> { std::size_t operator()(const Figure&) const; };
bool operator==(const Figure& lhs, const Figure& rhs);

Figure make_figure(const std::vector<std::string>& text);
std::unordered_set<Figure> make_figures(const std::vector<std::string>& text);
std::vector<Figure> read_figures(std::istream& str);
