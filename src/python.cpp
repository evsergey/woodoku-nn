#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/numpy.hpp>
#include <boost/tokenizer.hpp>
#include "figure.h"
#include "field.h"

using namespace boost::python;
namespace np = boost::python::numpy;

namespace pywood
{
    template<typename Container>
    auto to_stl(const object& iterable)
    {
        using T = typename Container::value_type;
        return Container(stl_input_iterator<T>(iterable), stl_input_iterator<T>());
    }

    template<class T>
    std::string to_string(const T& x)
    {
        std::stringstream ss;
        ss << x;
        return ss.str();
    }

    auto make_figures(const std::string& str)
    {
        boost::char_separator<char> sep("\n");
        boost::tokenizer<boost::char_separator<char>> tokens(str, sep);
        std::vector<std::string> lines(tokens.begin(), tokens.end());
        auto result = ::make_figures(lines);
        return std::vector<Figure>(result.begin(), result.end());
    }

    auto read_figures(const std::string& str)
    {
        std::istringstream ss(str);
        return ::read_figures(ss);
    }

    auto parse_figure(const std::string& str)
    {
        boost::char_separator<char> sep("\n");
        boost::tokenizer<boost::char_separator<char>> tokens(str, sep);
        std::vector<std::string> lines(tokens.begin(), tokens.end());
        return std::make_shared<Figure>(make_figure(lines));
    }

    void add_figure(Field& self, const Figure& figure, size_t row, size_t col)
    {
        self.add(figure, row, 1 << col);
    }

    size_t add_random(Field& self, const object& figures, int seed)
    {
        if (seed == 0)
        {
            std::random_device dev;
            seed = (int)dev();
        }
        std::default_random_engine rng(seed);
        extract<Figure> ex(figures);
        if (ex.check())
        {
            const Figure& f = ex();
            return self.add_random(f, rng) * f.weight;
        }
        else
        {
            size_t score = 0;
            for (stl_input_iterator<Figure> it(figures), it_end; it != it_end; ++it)
                if (!self.add_random(*it, rng))
                    break;
                else
                    score += it->weight;
            return score;
        }
    }

    tuple get_all_next(const Field& self, const object& iterable)
    {
        size_t score;
        auto figures = to_stl<std::vector<Figure>>(iterable);
        auto fields = self.get_all_next(figures.at(0), figures.at(1), figures.at(2), score);
        return boost::python::make_tuple(std::move(fields), score);
    }

    tuple get_all_next_2(const Field& self, const object& iterable, bool with_choices)
    {
        if (!with_choices)
            return get_all_next(self, iterable);
        size_t score;
        auto figures = to_stl<std::vector<Figure>>(iterable);
        std::vector<Choice> choices;
        auto fields = self.get_all_next(figures.at(0), figures.at(1), figures.at(2), score, &choices);
        return boost::python::make_tuple(std::move(fields), std::move(choices), score);
    }

    std::string print_choice(const Field& self, const object& iterable, Choice choice)
    {
        std::ostringstream ss;
        auto figures = to_stl<std::vector<Figure>>(iterable);
        self.print_choice(ss, { figures.at(0), figures.at(1), figures.at(2) }, choice);
        return ss.str();
    }

    template<class T>
    void copy_to_numpy(np::ndarray& out, const std::vector<Field>& fields)
    {
        T* data = reinterpret_cast<T*>(out.get_data());
        Field::copy_to(fields, data);
    }

    template<class T>
    bool check_type(np::dtype dtype)
    {
        return dtype == np::dtype::get_builtin<T>();
    }

    np::ndarray to_numpy(const std::vector<Field>& fields, object dtype)
    {
        tuple shape = make_tuple(fields.size(), 9, 9);
        np::dtype dt(dtype);
        auto result = np::zeros(shape,dt);
#define CHECK_AND_COPY(T) \
        if (check_type<T>(dt)) \
            copy_to_numpy<T>(result, fields); \
        else
#define CHECK_AND_COPY_I(bits) \
            CHECK_AND_COPY(int##bits##_t) \
            CHECK_AND_COPY(uint##bits##_t)
        CHECK_AND_COPY(double)
        CHECK_AND_COPY(float)
        CHECK_AND_COPY_I(8)
        CHECK_AND_COPY_I(16)
        CHECK_AND_COPY_I(32)
        CHECK_AND_COPY_I(64)
            throw std::runtime_error("Bad dtype");
        return result;
    }

    void shrink(std::vector<Field>& fields, size_t expected_size)
    {
        std::random_device dev;
        std::default_random_engine rng(dev());
        Field::random_shrink(fields, expected_size, rng);
    }
}

BOOST_PYTHON_MODULE(pywood)
{
    np::initialize();
    class_<Figure>("Figure", no_init)
        .def("__init__", make_constructor(&pywood::parse_figure))
        .def_readonly("nrows", &Figure::nrows)
        .def_readonly("ncols", &Figure::ncols)
        .def_readonly("weight", &Figure::weight)
        .def("__str__", &pywood::to_string<Figure>)
        .def("__repr__", &pywood::to_string<Figure>)
        .def("make", &pywood::make_figures, return_value_policy<return_by_value>())
        .staticmethod("make")
        .def("read", &pywood::read_figures, return_value_policy<return_by_value>())
        .staticmethod("read")
        ;
    class_<std::vector<Figure>>("Figures")
        .def(vector_indexing_suite<std::vector<Figure>>())
        ;
    class_<Field>("Field")
        .def(init<std::string>())
        .def("__str__", &pywood::to_string<Field>)
        .def("__repr__", &pywood::to_string<Field>)
        .add_property("weight", &Field::weight)
        .def("add", &pywood::add_figure)
        .def("add_random", &pywood::add_random)
        .def("add_random", +[](Field& self, const object& figures) { pywood::add_random(self, figures, 0); })
        .def("count_placements", &Field::count_placements)
        .def("can_place", &Field::has_placements)
        .def("get_all_next", &pywood::get_all_next)
        .def("get_all_next", &pywood::get_all_next_2)
        .def("copy", +[](const Field& field) { return field; }, return_value_policy<return_by_value>())
        .def("print_choice", &pywood::print_choice)
        ;
    class_<std::vector<Field>>("Fields")
        .def(vector_indexing_suite<std::vector<Field>>())
        .def("clear", +[](std::vector<Field>& fields) { fields.clear(); })
        .def("to_numpy", &pywood::to_numpy)
        .def("to_numpy", +[](const std::vector<Field>& fields) { return pywood::to_numpy(fields, np::dtype::get_builtin<double>()); })
        .def("shrink", &pywood::shrink)
        ;
    class_<Choice>("Choice")
        .def("__str__", &::pywood::to_string<Choice>)
        ;
    class_<std::vector<Choice>>("Choices")
        .def(vector_indexing_suite<std::vector<Choice>>())
        .def("clear", +[](std::vector<Choice>& choices) { choices.clear(); })
        ;
}
