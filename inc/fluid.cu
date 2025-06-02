#pragma once

#include <omp.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "SDL_rect.h"

#include "config.hpp"
#include "helper.cu"

template <int H, int W>
class Fluid {
 private:
  __device__ __host__ inline float interpolate_smoke(float x, float y) const;
  __device__ __host__ inline float get_general_velocity_y(float x,
                                                          float y) const;
  __device__ __host__ inline float get_general_velocity_x(float x,
                                                          float y) const;
  __device__ __host__ inline bool index_is_valid(int i, int j) const;
  __device__ __host__ inline bool is_valid_fluid(int i, int j) const;
  __device__ __host__ inline Vector2d<float> get_center_position(int i,
                                                                 int j) const;
  __device__ __host__ inline Vector2d<float> get_u_position(int i, int j) const;
  __device__ __host__ inline Vector2d<float> get_v_position(int, int j) const;

  inline void zero_pressure();

  inline void apply_external_forces(float d_t);
  inline void apply_projection(float d_t);
  inline void apply_smoke_advection(float d_t);
  inline void apply_velocity_advection(float d_t);
  inline void apply_extrapolation();
  inline void decay_smoke(float d_t);

 public:
  const float g = PHYSICS_G;
  const float o;
  const int cell_size;
  const int n;
  float pressure[W][H];
  float vel_x[W][H];
  float vel_y[W][H];
  float smoke[W][H];
  float vel_x_buffer[W][H];
  float vel_y_buffer[W][H];
  float smoke_buffer[W][H];
  int is_solid[W][H];
  int total_s[W][H];
  Fluid<H, W>* device_fluid;
  dim3 kernel_grid_dim;
  dim3 kernel_block_dim;

  Fluid(float o, int n, int cell_size);
  ~Fluid();

  // getters
  __device__ __host__ inline float get_divergence(int i, int j) const;

  __device__ __host__ inline bool is_edge(int i, int j) const;

  __device__ __host__ inline Vector2d<float> get_general_velocity(
      float x,
      float y) const;
  __device__ __host__ inline Vector2d<float> get_vertical_edge_velocity(
      int i,
      int j) const;
  __device__ __host__ inline Vector2d<float> get_horizontal_edge_velocity(
      int i,
      int j) const;
  __device__ __host__ inline std::array<SDL_Point, TRACE_LENGTH>
  trace(int i, int j, float d_t) const;

  __device__ __host__ inline void apply_external_forces_at(int i,
                                                           int j,
                                                           float d_t);
  __device__ __host__ inline void apply_projection_at(int i, int j, float d_t);
  __device__ __host__ inline void apply_velocity_advection_at(int i,
                                                              int j,
                                                              float d_t);
  __device__ __host__ inline void update_pressure_at(int i,
                                                     int j,
                                                     float velocity_diff,
                                                     float d_t);
  __device__ __host__ inline void apply_smoke_advection_at(int i,
                                                           int j,
                                                           float d_t);
  __device__ __host__ inline void update_smoke_advection_at(int i,
                                                            int j,
                                                            float d_t);
  __device__ __host__ inline void update_velocity_advection_at(int i,
                                                               int j,
                                                               float d_t);
  __device__ __host__ inline void apply_extrapolation_at(int i, int j);
  __device__ __host__ inline void decay_smoke_at(int i, int j, float d_t);
  __device__ __host__ inline void zero_pressure_at(int i, int j);
  inline void update(float d_t);
};

template <int H, int W>
__device__ __host__ inline std::array<SDL_Point, TRACE_LENGTH>
Fluid<H, W>::trace(int i, int j, float d_t) const {
  Vector2d<float> position = this->get_center_position(i, j);
  std::array<SDL_Point, TRACE_LENGTH> trace_points;
  trace_points[0] = {static_cast<int>(position.get_x()),
                     H - 1 - static_cast<int>(position.get_y())};
  for (int k = 1; k < TRACE_LENGTH; k++) {
    auto x = position.get_x();
    auto y = position.get_y();
    Vector2d<float> velocity = this->get_general_velocity(x, y);
    position = position + velocity * d_t;
    trace_points[k] = {static_cast<int>(position.get_x()),
                       H - 1 - static_cast<int>(position.get_y())};
  }
  return trace_points;
}

template <int H, int W>
__device__ __host__ inline bool Fluid<H, W>::is_edge(int i, int j) const {
  return i == 0 || j == 0 || i == W - 1 || j == H - 1;
}

template <int H, int W>
Fluid<H, W>::Fluid(float o, int n, int cell_size)
    : o(o), n(n), cell_size(cell_size) {
  int grid_x = std::ceil(static_cast<float>(W) / BLOCK_SIZE_X);
  int grid_y = std::ceil(static_cast<float>(H) / BLOCK_SIZE_Y);
  this->kernel_grid_dim = dim3(grid_x, grid_y, 1);
  this->kernel_block_dim = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  cudaMalloc(&this->device_fluid, sizeof(Fluid<H, W>));
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      this->is_solid[i][j] =
          (i == 0 or j == 0 or j == H - 1
#if ENABLE_RIGHT_WALL
           or i == W - 1
#endif
           or
           (ENABLE_CIRCLE and std::sqrt(std::pow((i - CIRCLE_POSITION_X), 2) +
                                        std::pow((j - CIRCLE_POSITION_Y), 2)) <
                                  CIRCLE_RADIUS or
            (i < PIPE_LENGTH && (j == H / 2 - PIPE_HEIGHT / 2 - 1 or
                                 j == H / 2 + PIPE_HEIGHT / 2 + 1))));
    }
  }
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      if (index_is_valid(i - 1, j) and is_solid[i - 1][j] == 0) {
        this->total_s[i][j]++;
      }
      if (index_is_valid(i + 1, j) and is_solid[i + 1][j] == 0) {
        this->total_s[i][j]++;
      }
      if (index_is_valid(i, j - 1) and is_solid[i][j - 1] == 0) {
        this->total_s[i][j]++;
      }
      if (index_is_valid(i, j + 1) and is_solid[i][j + 1] == 0) {
        this->total_s[i][j]++;
      }
    }
  }
  cudaMemcpy(this->device_fluid, this, sizeof(Fluid<H, W>),
             cudaMemcpyHostToDevice);
}

template <int H, int W>
Fluid<H, W>::~Fluid<H, W>() {
  cudaFree(this->device_fluid);
}

template <int H, int W>
__device__ __host__ inline float Fluid<H, W>::get_divergence(int i,
                                                             int j) const {
  auto u = this->vel_x[i][j];
  auto v = this->vel_y[i][j];
  auto top_v = this->vel_y[i][j + 1];
  auto right_u = this->vel_x[i + 1][j];

  auto divergence = right_u - u + top_v - v;

  return divergence;
}

template <int H, int W>
__global__ void zero_pressure_kernel(Fluid<H, W>* device_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->zero_pressure_at(i, j);
}

template <int H, int W>
__device__ __host__ inline void Fluid<H, W>::zero_pressure_at(int i, int j) {
  this->pressure[i][j] = 0;
}

template <int H, int W>
inline void Fluid<H, W>::zero_pressure() {
  zero_pressure_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(
      this->device_fluid);
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::update_pressure_at(int i, int j, float velocity_diff, float d_t) {
  this->pressure[i][j] += velocity_diff * FLUID_DENSITY * CELL_SIZE / d_t;
}

template <int H, int W>
__device__ __host__ inline void Fluid<H, W>::apply_projection_at(int i,
                                                                 int j,
                                                                 float d_t) {
  if (this->is_solid[i][j]) {
    return;
  }

  auto u = this->vel_x[i][j];
  auto v = this->vel_y[i][j];
  auto top_v = this->vel_y[i][j + 1];
  auto right_u = this->vel_x[i + 1][j];

  auto divergence = right_u - u + top_v - v;
  auto s = this->total_s[i][j];
  auto velocity_diff = this->o * (divergence / s);

#if ENABLE_PRESSURE
  if (i >= SMOKE_LENGTH + 1 or j >= H / 2 + PIPE_HEIGHT / 2 or
      j <= H / 2 - PIPE_HEIGHT / 2)
    this->update_pressure_at(i, j, velocity_diff, d_t);
#endif

  if (not this->is_solid[i - 1][j]) {
    this->vel_x[i][j] += velocity_diff;
  }

  if (not this->is_solid[i + 1][j]) {
    this->vel_x[i + 1][j] -= velocity_diff;
  }

  if (not this->is_solid[i][j - 1]) {
    this->vel_y[i][j] += velocity_diff;
  }

  if (not this->is_solid[i][j + 1]) {
    this->vel_y[i][j + 1] -= velocity_diff;
  }
}

template <int H, int W>
__global__ void apply_projection_even_kernel(Fluid<H, W>* fluid, float d_t) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + (j % 2);
  if (i >= W - 1 or j >= H - 1) {
    return;
  }
  fluid->apply_projection_at(i, j, d_t);
}

template <int H, int W>
__global__ void apply_projection_odd_kernel(Fluid<H, W>* fluid, float d_t) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((j + 1) % 2);
  if (i >= W - 1 or j >= H - 1) {
    return;
  }
  fluid->apply_projection_at(i, j, d_t);
}

template <int H, int W>
inline void Fluid<H, W>::apply_projection(float d_t) {
  int grid_x = std::ceil(static_cast<float>(W) / BLOCK_SIZE_X / 2);
  int grid_y = std::ceil(static_cast<float>(H) / BLOCK_SIZE_Y);
  auto grid_dim = dim3(grid_x, grid_y);
  auto block_dim = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

  for (int _ = 0; _ < this->n; _++) {
    apply_projection_even_kernel<<<grid_dim, block_dim>>>(
        this->device_fluid, d_t);
    apply_projection_odd_kernel<<<grid_dim, block_dim>>>(
        this->device_fluid, d_t);
  }
}

template <int H, int W>
__global__ void apply_external_forces_kernel(Fluid<H, W>* fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  fluid->apply_external_forces_at(i, j, d_t);
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::apply_external_forces_at(int i, int j, float d_t) {
  if (i <= SMOKE_LENGTH and i != 0 && j >= H / 2 - PIPE_HEIGHT / 2 &&
      j <= H / 2 + PIPE_HEIGHT / 2) {
    this->smoke[i][j] = WIND_SMOKE;
    this->vel_x[i][j] = WIND_SPEED;
  }
  this->vel_y[i][j] += PHYSICS_G * d_t;
}

template <int H, int W>
inline void Fluid<H, W>::apply_external_forces(float d_t) {
  apply_external_forces_kernel<<<this->kernel_grid_dim,
                                 this->kernel_block_dim>>>(this->device_fluid,
                                                           d_t);
}

template <int H, int W>
__device__ __host__ inline bool Fluid<H, W>::index_is_valid(int i,
                                                            int j) const {
  return i < W and j < H and i >= 0 and j >= 0;
}

template <int H, int W>
__device__ __host__ inline bool Fluid<H, W>::is_valid_fluid(int i,
                                                            int j) const {
  return index_is_valid(i, j) and not this->is_solid[i][j];
}

template <int H, int W>
__device__ __host__ inline Vector2d<float>
Fluid<H, W>::get_vertical_edge_velocity(int i, int j) const {
  auto u = this->vel_x[i][j];

  auto avg_v = this->vel_y[i][j];
  int count = 1;

  if (is_valid_fluid(i - 1, j + 1)) {
    avg_v += this->vel_y[i - 1][j + 1];
    count++;
  }

  if (is_valid_fluid(i, j + 1)) {
    avg_v += this->vel_y[i][j + 1];
    count++;
  }

  if (is_valid_fluid(i - 1, j)) {
    avg_v += this->vel_y[i - 1][j];
    count++;
  }

  avg_v /= count;

  return Vector2d<float>(u, avg_v);
}

template <int H, int W>
__device__ __host__ inline Vector2d<float>
Fluid<H, W>::get_horizontal_edge_velocity(int i, int j) const {
  auto v = this->vel_y[i][j];

  float avg_u = this->vel_x[i][j];
  int count = 1;

  if (is_valid_fluid(i + 1, j)) {
    avg_u += this->vel_x[i + 1][j];
    count++;
  }

  if (is_valid_fluid(i, j - 1)) {
    avg_u += this->vel_x[i][j - 1];
    count++;
  }

  if (is_valid_fluid(i + 1, j - 1)) {
    avg_u += this->vel_x[i + 1][j - 1];
    count++;
  }

  avg_u /= count;

  return Vector2d<float>(avg_u, v);
}

template <int H, int W>
__device__ __host__ inline float Fluid<H, W>::get_general_velocity_y(
    float x,
    float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  if (not this->is_valid_fluid(i, j)) {
    return 0;
  }

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_v = 0;

  // take average with the left cell
  if (in_x < this->cell_size / 2.0) {
    float d_x = this->cell_size / 2.0 - in_x;
    float w_x = 1 - d_x / this->cell_size;
    float w_y = 1 - in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_v += w_y * w_x * this->vel_y[i][j];
    }

    if (this->is_valid_fluid(i - 1, j)) {
      avg_v += w_y * (1 - w_x) * this->vel_y[i - 1][j];
    }

    if (this->is_valid_fluid(i - 1, j + 1)) {
      avg_v += (1 - w_y) * (1 - w_x) * this->vel_y[i - 1][j + 1];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_v += (1 - w_y) * w_x * this->vel_y[i][j + 1];
    }
  }
  // take average with the right cell
  else {
    float d_x = in_x - this->cell_size / 2.0;
    float w_x = 1 - d_x / this->cell_size;
    float w_y = 1 - in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_v += w_y * w_x * this->vel_y[i][j];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_v += (1 - w_y) * w_x * this->vel_y[i][j + 1];
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      avg_v += (1 - w_y) * (1 - w_x) * this->vel_y[i + 1][j + 1];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_v += w_y * (1 - w_x) * this->vel_y[i + 1][j];
    }
  }

  return avg_v;
}

template <int H, int W>
__device__ __host__ inline float Fluid<H, W>::get_general_velocity_x(
    float x,
    float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  if (not this->is_valid_fluid(i, j)) {
    return 0;
  }

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  float avg_u = 0;

  // take average with the bottom cell
  if (in_y <= this->cell_size / 2.0) {
    float d_y = this->cell_size / 2.0 - in_y;
    float w_x = 1 - in_x / this->cell_size;
    float w_y = 1 - d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_u += w_y * w_x * this->vel_x[i][j];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_u += w_y * (1 - w_x) * this->vel_x[i + 1][j];
    }

    if (this->is_valid_fluid(i, j - 1)) {
      avg_u += (1 - w_y) * w_x * this->vel_x[i][j - 1];
    }

    if (this->is_valid_fluid(i + 1, j - 1)) {
      avg_u += (1 - w_y) * (1 - w_x) * this->vel_x[i + 1][j - 1];
    }
  }

  // take average with the top cell
  else {
    float d_y = in_y - this->cell_size / 2.0;
    float w_x = 1 - in_x / this->cell_size;
    float w_y = 1 - d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_u += w_y * w_x * this->vel_x[i][j];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_u += (1 - w_y) * w_x * this->vel_x[i][j + 1];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_u += w_y * (1 - w_x) * this->vel_x[i + 1][j];
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      avg_u += (1 - w_y) * (1 - w_x) * this->vel_x[i + 1][j + 1];
    }
  }

  return avg_u;
}

template <int H, int W>
__device__ __host__ inline Vector2d<float> Fluid<H, W>::get_general_velocity(
    float x,
    float y) const {
  float u = this->get_general_velocity_x(x, y);
  float v = this->get_general_velocity_y(x, y);
  return Vector2d<float>(u, v);
}

template <int H, int W>
__device__ __host__ inline Vector2d<float> Fluid<H, W>::get_center_position(
    int i,
    int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size,
                         (j + 0.5) * this->cell_size);
}

template <int H, int W>
__device__ __host__ inline Vector2d<float> Fluid<H, W>::get_u_position(
    int i,
    int j) const {
  return Vector2d<float>(i * this->cell_size, (j + 0.5) * this->cell_size);
}

template <int H, int W>
__device__ __host__ inline Vector2d<float> Fluid<H, W>::get_v_position(
    int i,
    int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size, j * this->cell_size);
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::apply_smoke_advection_at(int i, int j, float d_t) {
  Vector2d<float> current_pos = this->get_center_position(i, j);
  Vector2d<float> current_velocity =
      this->get_general_velocity(current_pos.get_x(), current_pos.get_y());
  auto prev_pos = current_pos - current_velocity * d_t;
  float new_smoke = interpolate_smoke(prev_pos.get_x(), prev_pos.get_y());
  this->smoke_buffer[i][j] = new_smoke;
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::update_smoke_advection_at(int i, int j, float d_t) {
  this->smoke[i][j] = this->smoke_buffer[i][j];
}

template <int H, int W>
__global__ void apply_smoke_advection_kernel(Fluid<H, W>* device_fluid,
                                             float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->apply_smoke_advection_at(i, j, d_t);
}

template <int H, int W>
__global__ void update_smoke_advection_kernel(Fluid<H, W>* device_fluid,
                                              float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->update_smoke_advection_at(i, j, d_t);
}

template <int H, int W>
inline void Fluid<H, W>::apply_smoke_advection(float d_t) {
  apply_smoke_advection_kernel<<<this->kernel_grid_dim,
                                 this->kernel_block_dim>>>(this->device_fluid,
                                                           d_t);
  update_smoke_advection_kernel<<<this->kernel_grid_dim,
                                  this->kernel_block_dim>>>(this->device_fluid,
                                                            d_t);
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::apply_velocity_advection_at(int i, int j, float d_t) {
  Vector2d<float> current_pos = this->get_u_position(i, j);
  Vector2d<float> current_velocity = this->get_vertical_edge_velocity(i, j);
  auto prev_pos = current_pos - current_velocity * d_t;
  float new_velocity =
      this->get_general_velocity_x(prev_pos.get_x(), prev_pos.get_y());
  this->vel_x_buffer[i][j] = new_velocity;

  current_pos = this->get_v_position(i, j);
  current_velocity = this->get_horizontal_edge_velocity(i, j);
  prev_pos = current_pos - current_velocity * d_t;
  new_velocity =
      this->get_general_velocity_y(prev_pos.get_x(), prev_pos.get_y());
  this->vel_y_buffer[i][j] = new_velocity;
}

template <int H, int W>
__device__ __host__ inline void
Fluid<H, W>::update_velocity_advection_at(int i, int j, float d_t) {
  this->vel_x[i][j] = this->vel_x_buffer[i][j];
  this->vel_y[i][j] = this->vel_y_buffer[i][j];
}

template <int H, int W>
__global__ void apply_velocity_advection_kernel(Fluid<H, W>* device_fluid,
                                                float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->apply_velocity_advection_at(i, j, d_t);
}

template <int H, int W>
__global__ void update_velocity_advection_kernel(Fluid<H, W>* device_fluid,
                                                 float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->update_velocity_advection_at(i, j, d_t);
}

template <int H, int W>
inline void Fluid<H, W>::apply_velocity_advection(float d_t) {
  apply_velocity_advection_kernel<<<this->kernel_grid_dim,
                                    this->kernel_block_dim>>>(
      this->device_fluid, d_t);
  update_velocity_advection_kernel<<<this->kernel_grid_dim,
                                     this->kernel_block_dim>>>(
      this->device_fluid, d_t);
}

template <int H, int W>
__device__ __host__ inline float Fluid<H, W>::interpolate_smoke(float x,
                                                                float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  Vector2d<int> indices_1(i, j);
  Vector2d<int> indices_2;
  Vector2d<int> indices_3;
  Vector2d<int> indices_4;

  float distance_sum = 0;
  float avg_smoke = 0;

  if (in_x < this->cell_size / 2.0 && in_y < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i - 1, j);
    indices_3 = Vector2d<int>(i, j - 1);
    indices_4 = Vector2d<int>(i - 1, j - 1);
  } else if (in_x < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i - 1, j);
    indices_3 = Vector2d<int>(i, j + 1);
    indices_4 = Vector2d<int>(i - 1, j + 1);
  } else if (in_y < this->cell_size / 2.0) {
    indices_2 = Vector2d<int>(i + 1, j);
    indices_3 = Vector2d<int>(i, j - 1);
    indices_4 = Vector2d<int>(i + 1, j - 1);
  } else {
    indices_2 = Vector2d<int>(i + 1, j);
    indices_3 = Vector2d<int>(i, j + 1);
    indices_4 = Vector2d<int>(i + 1, j + 1);
  }

  Vector2d<float> pos_1 =
      get_center_position(indices_1.get_x(), indices_1.get_y());
  Vector2d<float> pos_2 =
      get_center_position(indices_2.get_x(), indices_2.get_y());
  Vector2d<float> pos_3 =
      get_center_position(indices_3.get_x(), indices_3.get_y());
  Vector2d<float> pos_4 =
      get_center_position(indices_4.get_x(), indices_4.get_y());

  auto distance_1 = get_distance(Vector2d<float>(x, y), pos_1);
  auto distance_2 = get_distance(Vector2d<float>(x, y), pos_2);
  auto distance_3 = get_distance(Vector2d<float>(x, y), pos_3);
  auto distance_4 = get_distance(Vector2d<float>(x, y), pos_4);

  distance_sum = distance_1 + distance_2 + distance_3 + distance_4;

  float inv1 = 1.0 / (distance_1 + 1e-6);
  float inv2 = 1.0 / (distance_2 + 1e-6);
  float inv3 = 1.0 / (distance_3 + 1e-6);
  float inv4 = 1.0 / (distance_4 + 1e-6);

  float sum_inv = inv1 + inv2 + inv3 + inv4;

  float w1 = inv1 / sum_inv;
  float w2 = inv2 / sum_inv;
  float w3 = inv3 / sum_inv;
  float w4 = inv4 / sum_inv;

  if (is_valid_fluid(indices_1.get_x(), indices_1.get_y())) {
    avg_smoke += w1 * this->smoke[indices_1.get_x()][indices_1.get_y()];
  }
  if (is_valid_fluid(indices_2.get_x(), indices_2.get_y())) {
    avg_smoke += w2 * this->smoke[indices_2.get_x()][indices_2.get_y()];
  }
  if (is_valid_fluid(indices_3.get_x(), indices_3.get_y())) {
    avg_smoke += w3 * this->smoke[indices_3.get_x()][indices_3.get_y()];
  }
  if (is_valid_fluid(indices_4.get_x(), indices_4.get_y())) {
    avg_smoke += w4 * this->smoke[indices_4.get_x()][indices_4.get_y()];
  }

  return avg_smoke;
}

// ? review this
// ? review logic
template <int H, int W>
__device__ __host__ inline void Fluid<H, W>::apply_extrapolation_at(int i,
                                                                    int j) {
  if (j == 0) {
    this->vel_x[i][j] = this->vel_x[i][j + 1];
    this->vel_y[i][j + 1] = 0;
  } else if (j == H - 1) {
    this->vel_x[i][j] = this->vel_x[i][j - 1];
  }
  if (i == 0) {
    this->vel_y[i][j] = this->vel_y[i + 1][j];
  } else if (i == W - 1) {
    this->vel_y[i][j] = this->vel_y[i - 1][j];
  }
}

template <int H, int W>
__global__ void apply_extrapolation_kernel(Fluid<H, W>* device_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->apply_extrapolation_at(i, j);
}

template <int H, int W>
inline void Fluid<H, W>::apply_extrapolation() {
  apply_extrapolation_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(
      this->device_fluid);
}

template <int H, int W>
__global__ void decay_smoke_kernel(Fluid<H, W>* device_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  device_fluid->decay_smoke_at(i, j, d_t);
}

template <int H, int W>
__device__ __host__ inline void Fluid<H, W>::decay_smoke_at(int i,
                                                            int j,
                                                            float d_t) {
#if ENABLE_SMOKE_DECAY
  this->smoke[i][j] = max(smoke - SMOKE_DECAY_RATE * d_t, 0.0);
#endif
}

template <int H, int W>
inline void Fluid<H, W>::decay_smoke(float d_t) {
  decay_smoke_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(
      this->device_fluid, d_t);
}

// ? put the whole thing into a graph
template <int H, int W>
inline void Fluid<H, W>::update(float d_t) {
  this->apply_external_forces(d_t);
#if ENABLE_PRESSURE
  this->zero_pressure();
#endif
  this->apply_projection(d_t);
  cudaMemcpyAsync(this->pressure, this->device_fluid->pressure,
                  sizeof(float) * H * W, cudaMemcpyDeviceToHost);
  this->apply_extrapolation();
  this->apply_velocity_advection(d_t);
  cudaMemcpyAsync(this->vel_x, this->device_fluid->vel_x, sizeof(float) * H * W,
                  cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(this->vel_y, this->device_fluid->vel_y, sizeof(float) * H * W,
                  cudaMemcpyDeviceToHost);
#if ENABLE_SMOKE
  if (WIND_SMOKE != 0) {
    this->apply_smoke_advection(d_t);
    this->decay_smoke(d_t);
  }
#endif
  cudaMemcpyAsync(this->smoke, this->device_fluid->smoke, sizeof(float) * H * W,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}