# Fluid Simulation Report

## Modes

The simulation can be run in four main visualization modes

- none
- smoke
- pressure
- pressure + smoke

Choosing either mode or changing any configurations from the config.hpp file will require a recompile since it was decided to implement mode changes using macros to achieve minimum runtime overhead. It's also possible to display additional information like velocity vectors and path lines.

| Mode                            |  Picture             | Mode                      | Picture              |
|:-------------------------------:|:--------------------:|:-------------------------:|:--------------------:|
| Smoke + Pressure                | ![mode](image.png)   | Smoke                     | ![mode](image-4.png) |
| Pressure                        | ![mode](image-3.png) | None + Path lines enabled | ![mode](image-2.png) |
| None + Velocity vectors enabled | ![mode](image-1.png) |                           |                      |

## Performance

The program logs the FPS and the time it took to complete a step in the simulation in milliseconds each time it completes a step. The time logged also includes the time it took to update the graphics since we were able to parallelize some part of the graphics.

\
\
\
To build the program GCC 13.3.0 is used with the following flags.

```shell
-O3 -march=native -fopenmp
```

\
Performance on 800x400 simulation with pressure and smoke enabled
\
Hardware: AMD Ryzen 7 5800H (16) @ 4.46 GHz

| P  | FPS | T(us) | W      | S    | E    | R    |
|:--:|:---:|:-----:|:------:|:----:|:----:|:----:|
| 1  | 14  | 71627 | 71265  | 1.00 | 1.00 | 1.00 |
| 2  | 23  | 43669 | 86878  | 1.64 | 0.82 | 1.22 |
| 3  | 32  | 31041 | 92607  | 2.31 | 0.77 | 1.3  |
| 4  | 41  | 24156 | 95986  | 2.97 | 0.74 | 1.35 |
| 5  | 50  | 20075 | 99625  | 3.57 | 0.71 | 1.40 |
| 6  | 56  | 17923 | 106762 | 4.00 | 0.67 | 1.50 |
| 7  | 63  | 15788 | 109658 | 4.50 | 0.65 | 1.54 |
| 8  | 66  | 14801 | 117186 | 4.84 | 0.60 | 1.64 |
| 9  | 49  | 20403 | 182295 | 3.51 | 0.39 | 2.56 |
| 10 | 53  | 18720 | 185856 | 3.83 | 0.38 | 2.61 |
| 11 | 57  | 17401 | 189641 | 4.12 | 0.37 | 2.66 |
| 12 | 60  | 16677 | 198338 | 4.29 | 0.36 | 2.78 |
| 13 | 63  | 15760 | 202890 | 4.54 | 0.35 | 2.85 |
| 14 | 63  | 15835 | 218296 | 4.52 | 0.32 | 3.06 |
| 15 | 62  | 16003 | 234705 | 4.48 | 0.30 | 3.29 |
| 16 | 49  | 20548 | 315750 | 3.49 | 0.28 | 4.43 |

![alt text](image-5.png)

Speedup peaks at 8 threads since my CPU has 8 cores. Even though it has 16 hardware threads, the program heavily utilizes floating-point arithmetics, and because FPUs are generally shared between hardware threads inside a core, creating more than 8 threads would not help with performance.

With the perf command on Linux we can check the cache miss rate to make sure that there are no problems with false sharing

| 8 threads                | single thread            |
|:------------------------:|:------------------------:|
| ![alt text](image-6.png) | ![alt text](image-7.png) |

The cache miss rate on both the single and multithreaded versions are identical, therefore there is no false sharing.

## Code explanation

The code structurally is split into two main classes `GraphicsHandler` and `Fluid`. In the main loop of the program the fluid object first gets updated and then the graphics get updated.

```cpp
// object initializations
Fluid fluid;
GraphicsHandler graphics;

// main loop
while (true) {
  fluid.update(d_t);
  graphics.update(fluid, d_t);
}
```

Inside of `fluid.update()` there are multiple stages. The job of each method is self explanatory.

```cpp
void Fluid::update() {
  this->apply_external_forces(d_t);
  this->apply_projection(d_t);
  this->extrapolate();
  this->apply_velocity_advection(d_t);
  this->apply_smoke_advection(d_t);
  this->decay_smoke(d_t);
}
```

There are a lot of places were the algorithm gets parallelized but let's take a look just inside one of these methods specifically `fluid.apply_projection()`

```cpp
void Fluid::apply_projection(float d_t)
  for (int _ = 0; _ < n; _++) {
#pragma omp parallel for schedule(static)
    for (int i = 1; i < W - 1; i++) {
      for (int j = i % 2 + 1; j < H - 1; j += 2) {
        this->step_projection(i, j, d_t);
      }
    }
#pragma omp parallel for schedule(static)
    for (int i = 1; i < W - 1; i++) {
      for (int j = (i + 1) % 2 + 1; j < H - 1; j += 2) {
        this->step_projection(i, j, d_t);
      }
    }
  }
```

It's a little unconventional to split a nested `for` loop like this, but doing so will nearly double the speed of the program. It all has to do with how cache is handled and how projection is actually performed.
In the **step_projection** method, at each call, we use the left, right, bottom, and top adjacent values to (i, j). Therefore, after the first sweep, the second sweep will need the values that were loaded in the previous sweep.
By putting all the calls into a single `for` loop, we have a high probability of evicting older values in the same sweep from the cache, which will then be requested again by the next row sweep, wasting time on fetching those values again from RAM.
By splitting the `for` loop into two, we can decrease the chances of evicting old values during a row sweep.
