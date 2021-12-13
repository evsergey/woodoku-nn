cmake_minimum_required(VERSION 3.5)
project(woodoku VERSION 1.0.0.0 LANGUAGES CXX)
find_package(PythonLibs)
find_package(Boost CONFIG COMPONENTS python numpy)
find_package(Torch)

set(CMAKE_CXX_STANDARD 20)
include_directories(include)

add_library(woodoku STATIC src/field.cpp src/figure.cpp include/field.h include/figure.h)
set_property(TARGET woodoku PROPERTY POSITION_INDEPENDENT_CODE ON)

if(Boost_FOUND AND PythonLibs_FOUND)
    add_library(pywood SHARED src/python.cpp)
    if (WIN32)
        set_target_properties(pywood PROPERTIES SUFFIX ".pyd")
    endif()
    set_target_properties(pywood PROPERTIES PREFIX "")
    target_compile_definitions(pywood PRIVATE BOOST_PYTHON_STATIC_LIB BOOST_NUMPY_STATIC_LIB)
    target_include_directories(pywood PRIVATE ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(pywood PRIVATE woodoku Boost::python Boost::numpy ${PYTHON_LIBRARIES})
endif()

if(Torch_FOUND)
    add_library(woodoku-solver SHARED src/solver.cpp include/solver.h)
    set_property(TARGET woodoku-solver PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET woodoku-solver PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS 1)
    target_link_libraries(woodoku-solver PUBLIC woodoku ${TORCH_LIBRARIES})
    set_property(TARGET woodoku-solver PROPERTY CXX_STANDARD 14)
    target_compile_options(woodoku-solver PRIVATE ${TORCH_CXX_FLAGS})
    if(MSVC)
      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
      add_custom_command(TARGET woodoku-solver
                         POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy_if_different
                         ${TORCH_DLLS}
                         $<TARGET_FILE_DIR:woodoku-solver>)
    endif()

    add_executable(woodoku_test src/woodoku.cpp)
    target_link_libraries(woodoku_test PRIVATE woodoku-solver)
    add_custom_command(
      TARGET woodoku_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${CMAKE_CURRENT_SOURCE_DIR}/figures.txt 
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(
      TARGET woodoku_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${CMAKE_CURRENT_SOURCE_DIR}/trained/scripted_3000.torch 
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/model.torch)
endif()
