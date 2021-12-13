#pragma once
#include "figure.h"
#include <random>

struct Choice
{
    uint32_t code;
    bool operator == (Choice c) const { return c.code == code; }
};

std::ostream& operator<< (std::ostream& str, const Choice& choice);

template<> struct std::hash<class Field> { std::size_t operator()(const Field&) const; };

class Field
{
    std::array<int32_t, 9> rows;

    friend std::ostream& operator<< (std::ostream& str, const Field& field);
    friend struct std::hash<Field>;
    friend bool operator==(const Field& lhs, const Field& rhs);

public:
    Field();
    Field(const std::string& text);

    void add(const Figure& figure, size_t row, int32_t col_pow2);
    bool add_random(const Figure& figure, std::default_random_engine& rng);
    std::vector<Field> get_all_next(const std::array<std::reference_wrapper<const Figure>, 3>& tri_fig, size_t& score, std::vector<Choice>* choices = nullptr) const;
    auto get_all_next(const Figure& a, const Figure& b, const Figure& c, size_t& score, std::vector<Choice>* choices = nullptr) const
    {
        return get_all_next({ std::ref(a), std::ref(b), std::ref(c) }, score, choices);
    }

    static void random_shrink(std::vector<Field>& fields, size_t expected_size, std::default_random_engine& rng);
    template<class T>
    static void convert_to(const std::vector<Field>& fields, T* data, size_t start_index = 0, size_t length = 0);
    static void copy_to(const std::vector<Field>& fields, int32_t* data, size_t start_index = 0, size_t length = 0);
    size_t count_placements(const Figure& figure) const;
    bool has_placements(const Figure& figure) const;
    size_t weight() const;
    void print_placement(std::ostream& str, const Figure& figure, size_t row, size_t col, unsigned char ch = '+') const;
    void print_choice(std::ostream& str, const std::array<std::reference_wrapper<const Figure>, 3>& tri_fig, Choice choice) const;

private:
    template<class T>
    void find_all_placements(const Figure& figure, T callback) const;
    int32_t get_row_mask(const Figure& figure, size_t row) const;
};
