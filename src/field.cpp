#include "field.h"
#include <cassert>
#include <memory_resource>
#include <random>

std::ostream& operator<< (std::ostream& str, const Choice& choice)
{
    str << "[";
    bool first = true;
    for (int i = 2; i >= 0; --i)
    {
        size_t x = choice.code >> (10 * i);
        size_t n = (x >> 8) & 3;
        size_t row = (x >> 4) & 15;
        size_t col = x & 15;
        if (row == 15)
            continue;
        if (!first)
            str << ", ";
        else
            first = false;
        str << "(" << n << ", " << row << ", " << col << ")";
    }
    return str << "]";
}


Field::Field()
{
    rows.fill(~uint32_t(511));
}

Field::Field(const std::string& text) : Field()
{
    size_t pos = 0;
    for (auto ch : text)
        if (ch == 'x')
        {
            rows[pos / 9] |= 1 << (pos % 9);
            ++pos;
        } else if (ch == '.' && ++pos == 81)
            break;
}

void Field::add(const Figure& figure, size_t row, int32_t col_pow2)
{
    std::array<int32_t, 3> row3;
    for (size_t r = 0; r < figure.nrows; ++r)
    {
        assert((rows[r + row] & (figure.rows[r] * col_pow2)) == 0);
        rows[r + row] |= figure.rows[r] * col_pow2;
    }
    for (size_t r3 = 0; r3 < 3; ++r3)
        row3[r3] = rows[r3 * 3] & rows[r3 * 3 + 1] & rows[r3 * 3 + 2];
    int32_t all_raws = row3[0] & row3[1] & row3[2] & 511;
    for (size_t r = 0; r < figure.nrows; ++r)
        rows[r + row] = rows[r + row] * bool(uint32_t(rows[r + row]) + 1) | ~uint32_t(511);
    if (all_raws != 0)
        for (size_t r3 = 0; r3 < 3; ++r3)
        {
            int32_t r = row3[r3];
            r = (r & (r >> 1) & (r >> 2) & 0b001001001) * 7;
            r = ~(r | all_raws);
            for (size_t i = 0; i < 3; ++i)
                rows[r3 * 3 + i] &= r;
        }
    else
        for (size_t r3 = 0; r3 < 3; ++r3)
        {
            int32_t r = row3[r3];
            r = r & (r >> 1) & (r >> 2) & 0b001001001;
            if (r)
            {
                r = ~(r * 7);
                for (size_t i = 0; i < 3; ++i)
                    rows[r3 * 3 + i] &= r;
            }
        }
}

std::size_t std::hash<Field>::operator()(const Field& field) const
{
    size_t result = 14695981039346656037ULL;
    for (size_t i = 0; i < field.rows.size(); ++i)
        result = (result ^ size_t(field.rows[i])) * 1099511628211ULL;
    return result;
}

bool operator==(const Field& lhs, const Field& rhs)
{
    return std::equal(lhs.rows.begin(), lhs.rows.end(), rhs.rows.begin());
}

std::ostream& operator<< (std::ostream& str, const Field& field)
{
    for (auto row : field.rows)
    {
        assert((row | 511) == -1);
        for (int i = 0; i < 9; ++i, row >>= 1)
            str.put((row & 1) ? 'x' : '.');
        str << std::endl;
    }
    return str;
}

int32_t Field::get_row_mask(const Figure& figure, size_t row) const
{
    int32_t mask = 0;
    for (size_t r = 0; r < figure.nrows; ++r)
        for (int32_t frow = figure.rows[r], m = rows[row + r]; frow; frow >>= 1, m >>= 1)
            mask |= m * (frow & 1);
    return mask;
}

template<class T>
void Field::find_all_placements(const Figure& figure, T callback) const
{
    for (size_t row = 0; row < 10 - figure.nrows; ++row)
        for (int32_t mask = ~get_row_mask(figure, row); mask;)
        {
            auto col_pow2 = (int32_t)std::bit_floor((uint32_t)mask);
            mask ^= col_pow2;
            callback(row, col_pow2);
        }
}

size_t Field::count_placements(const Figure& figure) const
{
    size_t result = 32 * (10 - figure.nrows);
    for (size_t row = 0; row < 10 - figure.nrows; ++row)
        result -= std::popcount((uint32_t)get_row_mask(figure, row));
    return result;
}

bool Field::has_placements(const Figure& figure) const
{
    for (size_t row = 0; row < 10 - figure.nrows; ++row)
        if(get_row_mask(figure, row) != -1)
            return true;
    return false;
}

bool Field::add_random(const Figure& figure, std::default_random_engine& rng)
{
    std::array<std::pair<size_t, int32_t>, 81> placements;
    size_t nplacements = 0;
    find_all_placements(figure, [&placements, &nplacements](size_t row, int32_t col_pow2) {
        placements[nplacements++] = { row, col_pow2 };
        });
    if (nplacements == 0)
        return false;
    std::uniform_int_distribution<size_t> distr(0, nplacements - 1);
    auto [row, col_pow2] = placements[distr(rng)];
    add(figure, row, col_pow2);
    return true;
}

std::vector<Field> Field::get_all_next(const std::array<std::reference_wrapper<const Figure>, 3>& tri_fig, size_t& score, std::vector<Choice>* choices) const
{
    if (choices != nullptr)
        choices->clear();
    score = 0;
    std::array<size_t, 3> first_cnt;
    size_t sum_cnt = 0, min_cnt = 81, max_cnt = 0, prd_cnt = 1;
    for (size_t i = 0; i < first_cnt.size(); ++i)
    {
        sum_cnt += first_cnt[i] = count_placements(tri_fig[i]);
        min_cnt = std::min(min_cnt, first_cnt[i]);
        max_cnt = std::max(max_cnt, first_cnt[i]);
        prd_cnt *= first_cnt[i];
    }

    auto append_all = [tri_fig](const Field& src, Choice choice, size_t fig_num, std::pmr::unordered_map<Field, Choice>& results)
    {
        choice.code <<= 2;
        choice.code |= fig_num;
        choice.code <<= 8;
        src.find_all_placements(tri_fig[fig_num], [&results, &figure=tri_fig[fig_num].get(), &src, choice = choice.code](size_t row, int32_t col_pow2) {
            Field next = src;
            next.add(figure, row, col_pow2);
            results.emplace(next, Choice{choice | uint32_t(row << 4) | std::countr_zero((uint32_t)col_pow2)});
            });
    };

    auto make_result = [choices](std::pmr::unordered_map<Field, Choice>& res)
    {
        std::vector<Field> fields;
        fields.reserve(res.size());
        if (choices != nullptr)
        {
            choices->reserve(fields.size());
            for (const auto& [field, choice] : res)
            {
                fields.push_back(field);
                choices->push_back(choice);
            }
        }
        else
            for (const auto& [field, _] : res)
                fields.push_back(field);
        return fields;
    };

    std::pmr::monotonic_buffer_resource mem((sizeof(Field) + sizeof(Choice) * prd_cnt));

    std::pmr::unordered_map<Field, Choice> first_figure{ &mem };
    first_figure.reserve(sum_cnt);
    for (size_t i = 0; i < 3; ++i)
        append_all(*this, Choice{ (uint32_t)-1 }, i, first_figure);

    if (first_figure.empty())
        return {};

    std::pmr::unordered_map<Field, Choice> second_figure{ &mem };
    second_figure.reserve(first_figure.size() * (sum_cnt - min_cnt) / 2);
    for (const auto& [field, choice] : first_figure)
        for (int j = 0, i = (choice.code >> 8) & 3; j < 3; ++j)
            if (j != i)
                append_all(field, choice, j, second_figure);

    if (second_figure.empty())
    {
        for (const auto& [_, choice] : first_figure)
            score = std::max(score, tri_fig[(choice.code >> 8) & 3].get().weight);
        return {};
    }
    first_figure.clear();

    std::pmr::unordered_map<Field, Choice> third_figure{ &mem };
    third_figure.reserve(second_figure.size() * max_cnt / 3);
    for (const auto& [field, choice] : second_figure)
        append_all(field, choice, 3 - (choice.code >> 18) & 3 - (choice.code >> 8) & 3, third_figure);

    for (const Figure& fig : tri_fig)
        score += fig.weight;
    if (third_figure.empty())
    {
        size_t min_weight = 5;
        for (const Figure& fig : tri_fig)
            min_weight = std::min(min_weight, fig.weight);
        size_t found_min = 5;
        for (const auto& [_, choice] : second_figure)
        {
            int k = 3 - (choice.code >> 28) & 3 - (choice.code >> 18) & 3;
            if (const size_t w = tri_fig[k].get().weight; w < found_min && min_weight == (found_min = w))
                break;
        }
        score -= found_min;
        return {};
    }
    return make_result(third_figure);
}

size_t Field::weight() const
{
    size_t result = 0;
    for (auto row : rows)
        result += size_t(std::popcount((uint32_t)row) - (32 - 9));
    return result;
}

void Field::print_placement(std::ostream& str, const Figure& figure, size_t row, size_t col, unsigned char ch) const
{
    Field next = *this;
    Field clean;
    next.add(figure, row, 1 << col);
    clean.add(figure, row, 1 << col);
    bool disappeared = next.weight() < weight() + figure.weight;
    for (size_t r = 0; r < 9; ++r)
    {
        for (size_t c = 0; c < 9; ++c)
            if (rows[r] & (1 << c))
                str << 'x';
            else if (clean.rows[r] & (1 << c))
                str << ch;
            else
                str << '.';
        if (disappeared)
        {
            str << "   ";
            for (size_t c = 0; c < 9; ++c)
                if (next.rows[r] & (1 << c))
                    str << 'x';
                else if ((rows[r] | clean.rows[r]) & (1 << c))
                    str << '*';
                else
                    str << '.';
        }
        str << std::endl;
    }
}

void Field::print_choice(std::ostream& str, const std::array<std::reference_wrapper<const Figure>, 3>& tri_fig, Choice choice) const
{
    Field f = *this;
    for (int i = 2; i >= 0; --i)
    {
        size_t x = choice.code >> (10 * i);
        size_t n = (x >> 8) & 3;
        size_t row = (x >> 4) & 15;
        size_t col = x & 15;
        if (row == 15)
            break;
        if (i != 2)
            str << std::endl;
        f.print_placement(str, tri_fig[n], row, col, static_cast<unsigned char>('1' + n));
        f.add(tri_fig[n], row, 1 << col);
    }
}

template<class T>
void Field::copy_to(const std::vector<Field>& fields, T* data, size_t start_index, size_t length)
{
    if (length == 0)
        length = fields.size() - start_index;
    for (size_t i = 0; i < length; ++i)
        for (size_t row = 0; row < 9; ++row)
            for (size_t col = 0; col < 9; ++col)
                data[i * 81 + row * 9 + col] = T((fields[i + start_index].rows[row] >> col) & 1);
}

#define DEFCOPYTO(TYPE) \
    template void Field::copy_to<TYPE>(const std::vector<Field>& fields, TYPE* data, size_t start_index, size_t length);
DEFCOPYTO(double)
DEFCOPYTO(float)

#define DEFCOPYTOI(bits) \
    DEFCOPYTO(int##bits##_t) \
    DEFCOPYTO(uint##bits##_t)

DEFCOPYTOI(8)
DEFCOPYTOI(16)
DEFCOPYTOI(32)
DEFCOPYTOI(64)

void Field::random_shrink(std::vector<Field>& fields, size_t expected_size, std::default_random_engine& rng)
{
    if (expected_size >= fields.size())
        return;
    std::uniform_int_distribution<size_t> distr(0, fields.size() - 1);
    std::erase_if(fields, [&rng, &distr, expected_size](const auto&) { return distr(rng) > expected_size; });
    if (fields.size() > expected_size)
        fields.resize(expected_size);
}
