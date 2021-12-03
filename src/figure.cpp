#include "figure.h"
#include <cassert>
#include <numeric>
#include <string>

std::ostream& operator<< (std::ostream& str, const Figure& figure)
{
    for (auto row : figure.rows)
    {
        if (!row)
            break;
        for (int i = 0; row; ++i, row >>= 1)
            str.put((row & 1) ? 'x' : '.');
        str << std::endl;
    }
    return str;
}

std::size_t std::hash<Figure>::operator()(const Figure& figure) const
{
    int64_t result = 0;
    for (auto row : figure.rows)
        result = result * 32 + row;
    return std::hash<int64_t>{}(result);
}

bool operator==(const Figure& lhs, const Figure& rhs)
{
    return std::equal(lhs.rows.begin(), lhs.rows.end(), rhs.rows.begin());
}

namespace
{
    std::pair<size_t, size_t> get_transform(size_t width, size_t height, size_t row, size_t col, int rot, bool mirror)
    {
        if (mirror)
            col = width - 1 - col;
        switch (rot)
        {
        case 0: return { row, col };
        case 1: return { col, height - row - 1 };
        case 2: return { height - row - 1 , width - col - 1 };
        case 3: return { width - col - 1, row };
        default:
            assert(false);
            throw std::exception();
        }
    }
}

std::unordered_set<Figure> make_figures(const std::vector<std::string>& text, std::initializer_list<int> rots, std::initializer_list<bool> mirrors)
{
    std::unordered_set<Figure> figures;
    size_t height = text.size();
    assert(height != 0);
    size_t width = text[0].size();
    for (size_t i = 1; i < height; ++i)
        width = std::max(width, text[i].size());
    assert(width != 0);
    size_t weight = std::accumulate(
        text.begin(), text.end(), (size_t)0,
        [](size_t val, const std::string& str) { return val + std::count(str.begin(), str.end(), 'x'); }
    );
    for (bool mirror : mirrors)
        for (int rot: rots)
        {
            Figure figure
            {
                .rows = {},
                .weight = weight,
                .ncols = rot % 2 ? height : width,
                .nrows = rot % 2 ? width : height
            };
            for (size_t row = 0; row < height; ++row)
                for (size_t col = 0; col < text[row].size(); ++col)
                {
                    auto [r, c] = get_transform(width, height, row, col, rot, mirror);
                    figure.rows[r] |= (text[row][col] == 'x') << c;
                }
            figures.insert(figure);
        }
    return figures;
}

std::unordered_set<Figure> make_figures(const std::vector<std::string>& text)
{
    return make_figures(text, { 0, 1, 2, 3 }, { false, true });
}

Figure make_figure(const std::vector<std::string>& text)
{
    return *make_figures(text, { false }, { 0 }).begin();
}

std::vector<Figure> read_figures(std::istream& str)
{
    std::vector<Figure> result;
    std::vector<std::string> text;
    while (str)
    {
        std::string line;
        std::getline(str, line);
        if (!line.empty())
            text.push_back(line);
        else if (!text.empty())
        {
            auto tfigures = make_figures(text);
            result.insert(result.end(), tfigures.begin(), tfigures.end());
            text.clear();
        }
    }
    return result;
}
