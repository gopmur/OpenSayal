# OpenSayal

## Table of Content

- [About](#about)
- [Gallery](#gallery)
- [How to Build](#how-to-build)
- [Configuration](#configuration)

## About

OpenSayal is a lightweight fluid simulator accelerated using Cuda enabled Nvidia GPUs. OpenSayal can simulate incompressible fluids with zero viscosity. It supports gravity, drag and solids.

## Gallery

![image](https://github.com/user-attachments/assets/af31841f-5153-48ba-b8b3-65e701fe6f0b)

![image](https://github.com/user-attachments/assets/ed2dac9b-94e7-4d76-8ee6-ec8d34526b41)


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
- `gcc`
- `cuda`

```shell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release
```

Run the program.

```shell
cmake --build build --config Release
```

you can change build system by specifying your generator after -G.

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

## Configuration

Put a configuration file named `OpenSayal.conf.json` in the directory in which you wish to launch you program from. You can either copy the configuration provided below or configure from scratch.

```json
{
 "thread": {
  "cuda": {
   "block_size_x": 64,
   "block_size_y": 1
  }
 },
 "sim": {
  "height": 1080,
  "width": 1920,
  "cell_pixel_size": 1,
  "cell_size": 1,
  "enable_drain": false,
  "enable_pressure": true,
  "enable_smoke": true,
  "projection": {
   "n": 50,
   "o": 1.9
  },
  "enable_interactive": true,
  "wind_tunnel": {
   "pipe_height": 270,
   "smoke_length": 0,
   "speed": 0,
   "smoke": 1
  },
  "physics": {
   "g": -5
  },
  "time": {
   "d_t": 0.05,
   "enable_real_time": false,
   "real_time_multiplier": 1
  },
  "smoke": {
   "enable_decay": false,
   "decay_rate": 0
  },
  "obstacle": {
   "enable": false,
   "center_x": 960,
   "center_y": 540,
   "radius": 36
  }
 },
 "fluid": {
  "density": 1
 },
 "visual": {
  "arrows": {
   "color": {
    "r": 0,
    "g": 0,
    "b": 0,
    "a": 255
   },
   "enable": false,
   "distance": 20,
   "length_multiplier": 0.1,
   "disable_threshold": 0,
   "head_length": 5
  },
  "path_line": {
   "enable": false,
   "length": 20,
   "color": {
    "r": 255,
    "g": 255,
    "b": 255,
    "a": 255
   },
   "distance": 20
  }
 }
}
```

Every configuration key can be defined in a single line or can be defined in multiple levels. for example the following will do the exact same thing.

```json
{
  "sim": {
    "time": {
      "d_t": 0.05
    }
  }
}
```

```json
{
  "sim.time.d_t": 0.05
}
```

```json
{
  "sim": {
    "time.d_t": 0.05
  }
}
```

For more information on each setting read [here](/docs/configuration.md)
