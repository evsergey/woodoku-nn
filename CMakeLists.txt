cmake_minimum_required(VERSION 3.5)
project(woodoku VERSION 1.0.0.0 LANGUAGES CXX)
find_package(PythonLibs)
find_package(Boost CONFIG COMPONENTS python numpy)

set(CMAKE_CXX_STANDARD 20)
include_directories(include)

add_library(woodoku STATIC src/field.cpp src/figure.cpp)

if(Boost_FOUND AND PythonLibs_FOUND)
    add_library(pywood SHARED src/python.cpp)
    set_target_properties(pywood PROPERTIES SUFFIX ".pyd")
    target_compile_definitions(pywood PRIVATE BOOST_PYTHON_STATIC_LIB BOOST_NUMPY_STATIC_LIB)
    target_include_directories(pywood PRIVATE ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(pywood PRIVATE woodoku Boost::python Boost::numpy ${PYTHON_LIBRARIES})
endif()

add_executable(woodoku_test src/woodoku.cpp)
target_link_libraries(woodoku_test PRIVATE woodoku)
add_custom_command(
  TARGET woodoku_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${CMAKE_CURRENT_SOURCE_DIR}/figures.txt 
    $<TARGET_FILE_DIR:${PROJECT_NAME}>)