# woodoku-nn

Build steps:

    $ mkdir build
    $ cd build
    $ Torch_DIR=<path to libtorch> cmake -DCMAKE_CXX_COMPILER=<compiler> -DCUDA_TOOLKIT_ROOT_DIR=<path to CUDA> ..
    $ cmake --build .
