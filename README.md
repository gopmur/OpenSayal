# OpenSayal

## Table of Content

- [About](About)
- [How to Build](HowtoBuild)

## About

OpenSayal is a lightweight fluid simulator accelerated using Cuda enabled Nvidia GPUs. OpenSayal can simulate incompressible fluids with zero viscosity. It supports gravity, drag and solids.

## How to Build

### linux

Dependencies \
`git` \
`cmake` \
`ninja` or `make` \
`gcc` or `clang` \
`cuda`

Clone the repository

```shell
git clone https://github.com/gopmur/OpenSayal.git
```

Download dependencies

```shell
git submodule update --init --recursive
```

Build

```shell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

you can change CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to match your desired compiler or change build system by specifying your generator after -G.

Run

```shell
./build/OpenSayal
```
