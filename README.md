# OpenSayal

## Table of Content

- [About](#about)
- [How to Build](#how-to-build)

## About

OpenSayal is a lightweight fluid simulator accelerated using Cuda enabled Nvidia GPUs. OpenSayal can simulate incompressible fluids with zero viscosity. It supports gravity, drag and solids.

## How to Build

Clone the repository.

```shell
git clone https://github.com/gopmur/OpenSayal.git
```

Download dependencies.

```shell
git submodule update --init --recursive
```

### Linux

Build dependencies

- `cmake >= 4.0`
- `ninja` or `make`
- `gcc` or `clang`
- `cuda`

```shell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
```

Run the program.

```shell
cmake --build build --config Release
```

you can change CMAKE_C_COMPILER and CMAKE_CXX_COMPILER to match your desired compiler or change build system by specifying your generator after -G.

```shell
./build/OpenSayal
```

### Windows

Build dependencies

- `cmake >= 4.0`
- `Visual Studio`
- `cuda`

Enter Visual Studio Developer Powershell and run these commands in the root directory of the project.

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_HOST_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES="75;86;90"
```

```shell
cmake --build build --config Release
```

To run the program navigate to `build/Release` copy `OpenSayal.exe` to `build/lib/sdl/Release` and run `OpenSayal.exe` there.  
