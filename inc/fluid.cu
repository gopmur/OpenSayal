#pragma once

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "SDL_rect.h"

#include "config.hpp"
#include "helper.cu"

class Fluid {
 private:
  __device__ inline float interpolate_smoke(float x, float y) const;
  __device__ inline float get_general_velocity_y(float x, float y) const;
  __device__ inline float get_general_velocity_x(float x, float y) const;
  __device__ __host__ inline bool index_is_valid(int i, int j) const;
  __device__ inline bool is_valid_fluid(int i, int j) const;
  __device__ inline Vector2d<float> get_center_position(int i, int j) const;
  __device__ inline Vector2d<float> get_u_position(int i, int j) const;
  __device__ inline Vector2d<float> get_v_position(int, int j) const;

  inline void zero_pressure();

  inline void apply_external_forces(float d_t);
  inline void apply_projection(float d_t);
  inline void apply_smoke_advection(float d_t);
  inline void apply_velocity_advection(float d_t);
  inline void apply_extrapolation();
  inline void decay_smoke(float d_t);

  void alloc_device_memory();
  void init_device_memory();

 public:
  const int width;
  const int height;
  const float g = PHYSICS_G;
  const float o;
  const int cell_size;
  const int n;

  float min_pressure;
  float max_pressure;

  float* d_pressure;
  float* d_vel_x;
  float* d_vel_y;
  float* d_smoke;
  float* d_vel_x_buffer;
  float* d_vel_y_buffer;
  float* d_smoke_buffer;
  int* d_is_solid;
  int* d_total_s;

  Fluid* d_this;
  dim3 kernel_grid_dim;
  dim3 kernel_block_dim;

  Fluid(int width, int height, float o, int n, int cell_size);
  ~Fluid();

  __host__ __device__ inline int indx(int i, int j) const;

  __device__ inline float get_divergence(int i, int j) const;

  __device__ __host__ inline bool is_edge(int i, int j) const;

  __device__ inline Vector2d<float> get_general_velocity(float x,
                                                         float y) const;
  __device__ inline Vector2d<float> get_vertical_edge_velocity(int i,
                                                               int j) const;
  __device__ inline Vector2d<float> get_horizontal_edge_velocity(int i,
                                                                 int j) const;
  __device__ inline void trace(int i,
                               int j,
                               float d_t,
                               SDL_Point* trace_line) const;

  __device__ inline void apply_external_forces_at(int i, int j, float d_t);
  __device__ inline void apply_projection_at(int i, int j, float d_t);
  __device__ inline void apply_velocity_advection_at(int i, int j, float d_t);
  __device__ inline void update_pressure_at(int i,
                                            int j,
                                            float velocity_diff,
                                            float d_t);
  __device__ inline void apply_smoke_advection_at(int i, int j, float d_t);
  __device__ inline void update_smoke_advection_at(int i, int j, float d_t);
  __device__ inline void update_velocity_advection_at(int i, int j, float d_t);
  __device__ inline void apply_extrapolation_at(int i, int j);
  __device__ inline void decay_smoke_at(int i, int j, float d_t);
  __device__ inline void zero_pressure_at(int i, int j);
  inline void update(float d_t);
};

__device__ inline void Fluid::trace(int i,
                                    int j,
                                    float d_t,
                                    SDL_Point* trace_line) const {
  Vector2d<float> position = this->get_center_position(i, j);
  trace_line[0] = {
      static_cast<int>(round(position.get_x())),
      this->height - 1 - static_cast<int>(round(position.get_y()))};
  for (int k = 1; k < TRACE_LENGTH; k++) {
    auto x = position.get_x();
    auto y = position.get_y();
    Vector2d<float> velocity = this->get_general_velocity(x, y);
    position = position + velocity * d_t;
    trace_line[k] = {
        static_cast<int>(round(position.get_x())),
        this->height - 1 - static_cast<int>(round(position.get_y()))};
  }
}

__device__ __host__ inline bool Fluid::is_edge(int i, int j) const {
  return i == 0 || j == 0 || i == this->width - 1 || j == this->height - 1;
}

void Fluid::alloc_device_memory() {
  cudaMalloc(&this->d_this, sizeof(Fluid));
  cudaMalloc(&this->d_pressure, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_vel_x, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_vel_y, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_smoke, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_vel_x_buffer, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_vel_y_buffer, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_smoke_buffer, this->width * this->height * sizeof(float));
  cudaMalloc(&this->d_is_solid, this->width * this->height * sizeof(int));
  cudaMalloc(&this->d_total_s, this->width * this->height * sizeof(int));
}

void Fluid::init_device_memory() {
  int* is_solid =
      static_cast<int*>(std::malloc(this->height * this->width * sizeof(int)));
  int* total_s =
      static_cast<int*>(std::malloc(this->height * this->width * sizeof(int)));
  float* vel_x = static_cast<float*>(
      std::calloc(this->width * this->height, sizeof(float)));
  float* vel_y = static_cast<float*>(
      std::calloc(this->width * this->height, sizeof(float)));
  float* smoke = static_cast<float*>(
      std::calloc(this->width * this->height, sizeof(float)));

  for (auto i = 0; i < this->width; i++) {
    for (auto j = 0; j < this->height; j++) {
      is_solid[indx(i, j)] =
          (i == 0 or j == 0 or j == this->height - 1
#if ENABLE_RIGHT_WALL
           or i == W - 1
#endif
           or
           (ENABLE_CIRCLE and std::sqrt(std::pow((i - CIRCLE_POSITION_X), 2) +
                                        std::pow((j - CIRCLE_POSITION_Y), 2)) <
                                  CIRCLE_RADIUS or
            (i < PIPE_LENGTH &&
             (j == this->height / 2 - PIPE_HEIGHT / 2 - 1 or
              j == this->height / 2 + PIPE_HEIGHT / 2 + 1))));
    }
  }
  for (auto i = 0; i < this->width; i++) {
    for (auto j = 0; j < this->height; j++) {
      if (index_is_valid(i - 1, j) and is_solid[indx(i - 1, j)] == 0) {
        total_s[indx(i, j)]++;
      }
      if (index_is_valid(i + 1, j) and is_solid[indx(i + 1, j)] == 0) {
        total_s[indx(i, j)]++;
      }
      if (index_is_valid(i, j - 1) and is_solid[indx(i, j - 1)] == 0) {
        total_s[indx(i, j)]++;
      }
      if (index_is_valid(i, j + 1) and is_solid[indx(i, j + 1)] == 0) {
        total_s[indx(i, j)]++;
      }
    }
  }

  cudaMemcpy(this->d_is_solid, is_solid,
             this->width * this->height * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_total_s, total_s, this->width * this->height * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_vel_x, vel_x, this->width * this->height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_vel_y, vel_y, this->width * this->height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->d_smoke, smoke, this->width * this->height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_this, this, sizeof(Fluid), cudaMemcpyHostToDevice);

  std::free(is_solid);
  std::free(total_s);
  std::free(vel_x);
  std::free(vel_y);
  std::free(smoke);
}

__device__ __host__ inline int Fluid::indx(int i, int j) const {
  return (this->height - j - 1) * this->width + i;
}

Fluid::Fluid(int width, int height, float o, int n, int cell_size)
    : width(width), height(height), o(o), n(n), cell_size(cell_size) {
  int grid_x = std::ceil(static_cast<float>(width) / BLOCK_SIZE_X);
  int grid_y = std::ceil(static_cast<float>(height) / BLOCK_SIZE_Y);
  this->kernel_grid_dim = dim3(grid_x, grid_y, 1);
  this->kernel_block_dim = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  this->alloc_device_memory();
  this->init_device_memory();
}

Fluid::~Fluid() {
  cudaFree(this->d_this);
  cudaFree(this->d_pressure);
  cudaFree(this->d_vel_x);
  cudaFree(this->d_vel_y);
  cudaFree(this->d_smoke);
  cudaFree(this->d_vel_x_buffer);
  cudaFree(this->d_vel_y_buffer);
  cudaFree(this->d_smoke_buffer);
  cudaFree(this->d_is_solid);
  cudaFree(this->d_total_s);
}

__device__ inline float Fluid::get_divergence(int i, int j) const {
  auto u = this->d_vel_x[indx(i, j)];
  auto v = this->d_vel_y[indx(i, j)];
  auto top_v = this->d_vel_y[indx(i, j + 1)];
  auto right_u = this->d_vel_x[indx(i + 1, j)];

  auto divergence = right_u - u + top_v - v;

  return divergence;
}

__global__ void zero_pressure_kernel(Fluid* d_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->zero_pressure_at(i, j);
}

__device__ inline void Fluid::zero_pressure_at(int i, int j) {
  this->d_pressure[indx(i, j)] = 0;
}

inline void Fluid::zero_pressure() {
  zero_pressure_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(
      d_this);
}

__device__ inline void Fluid::update_pressure_at(int i,
                                                 int j,
                                                 float velocity_diff,
                                                 float d_t) {
  this->d_pressure[indx(i, j)] +=
      velocity_diff * FLUID_DENSITY * CELL_SIZE / d_t;
}

__device__ inline void Fluid::apply_projection_at(int i, int j, float d_t) {
  if (this->d_is_solid[indx(i, j)]) {
    return;
  }

  auto u = this->d_vel_x[indx(i, j)];
  auto v = this->d_vel_y[indx(i, j)];
  auto top_v = this->d_vel_y[indx(i, j + 1)];
  auto right_u = this->d_vel_x[indx(i + 1, j)];

  auto divergence = right_u - u + top_v - v;
  auto s = this->d_total_s[indx(i, j)];
  auto velocity_diff = this->o * (divergence / s);

#if ENABLE_PRESSURE
  if (i >= SMOKE_LENGTH + 1 or j >= this->height / 2 + PIPE_HEIGHT / 2 or
      j <= this->height / 2 - PIPE_HEIGHT / 2)
    this->update_pressure_at(i, j, velocity_diff, d_t);
#endif

  if (not this->d_is_solid[indx(i - 1, j)]) {
    this->d_vel_x[indx(i, j)] += velocity_diff;
  }

  if (not this->d_is_solid[indx(i + 1, j)]) {
    this->d_vel_x[indx(i + 1, j)] -= velocity_diff;
  }

  if (not this->d_is_solid[indx(i, j - 1)]) {
    this->d_vel_y[indx(i, j)] += velocity_diff;
  }

  if (not this->d_is_solid[indx(i, j + 1)]) {
    this->d_vel_y[indx(i, j + 1)] -= velocity_diff;
  }
}

__global__ void apply_projection_even_kernel(Fluid* d_fluid, float d_t) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + (j % 2);
  if (i >= d_fluid->width - 1 or j >= d_fluid->height - 1 or i <= 0 or j <= 0) {
    return;
  }
  d_fluid->apply_projection_at(i, j, d_t);
}

__global__ void apply_projection_odd_kernel(Fluid* d_fluid, float d_t) {
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((j + 1) % 2);
  if (i >= d_fluid->width - 1 or j >= d_fluid->height - 1 or i <= 0 or j <= 0) {
    return;
  }
  d_fluid->apply_projection_at(i, j, d_t);
}

inline void Fluid::apply_projection(float d_t) {
  int grid_x = std::ceil(static_cast<float>(this->width) / BLOCK_SIZE_X / 2);
  int grid_y = std::ceil(static_cast<float>(this->height) / BLOCK_SIZE_Y);
  auto grid_dim = dim3(grid_x, grid_y);
  auto block_dim = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

  for (int _ = 0; _ < this->n; _++) {
    apply_projection_even_kernel<<<grid_dim, block_dim>>>(d_this, d_t);
    apply_projection_odd_kernel<<<grid_dim, block_dim>>>(d_this, d_t);
  }
}

__global__ void apply_external_forces_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->apply_external_forces_at(i, j, d_t);
}

__device__ inline void Fluid::apply_external_forces_at(int i,
                                                       int j,
                                                       float d_t) {
  if (i <= SMOKE_LENGTH and i != 0 && j >= this->height / 2 - PIPE_HEIGHT / 2 &&
      j <= this->height / 2 + PIPE_HEIGHT / 2) {
    this->d_smoke[indx(i, j)] = WIND_SMOKE;
    this->d_vel_x[indx(i, j)] = WIND_SPEED;
  }
  if (this->d_vel_x[indx(i, j)] > 0) {
    this->d_vel_x[indx(i, j)] -= DRAG_COEFF * d_t;
  } else {
    this->d_vel_x[indx(i, j)] += DRAG_COEFF * d_t;
  }
  if (this->d_vel_y[indx(i, j)] > 0) {
    this->d_vel_y[indx(i, j)] -= DRAG_COEFF * d_t;
  } else {
    this->d_vel_y[indx(i, j)] += DRAG_COEFF * d_t;
  }
  this->d_vel_y[indx(i, j)] += PHYSICS_G * d_t;
}

inline void Fluid::apply_external_forces(float d_t) {
  apply_external_forces_kernel<<<this->kernel_grid_dim,
                                 this->kernel_block_dim>>>(d_this, d_t);
}

__device__ __host__ inline bool Fluid::index_is_valid(int i, int j) const {
  return i < this->width and j < this->height and i >= 0 and j >= 0;
}

__device__ inline bool Fluid::is_valid_fluid(int i, int j) const {
  return index_is_valid(i, j) and not this->d_is_solid[indx(i, j)];
}

__device__ inline Vector2d<float> Fluid::get_vertical_edge_velocity(
    int i,
    int j) const {
  auto u = this->d_vel_x[indx(i, j)];

  auto avg_v = this->d_vel_y[indx(i, j)];
  int count = 1;

  if (is_valid_fluid(i - 1, j + 1)) {
    avg_v += this->d_vel_y[indx(i - 1, j + 1)];
    count++;
  }

  if (is_valid_fluid(i, j + 1)) {
    avg_v += this->d_vel_y[indx(i, j + 1)];
    count++;
  }

  if (is_valid_fluid(i - 1, j)) {
    avg_v += this->d_vel_y[indx(i - 1, j)];
    count++;
  }

  avg_v /= count;

  return Vector2d<float>(u, avg_v);
}

__device__ inline Vector2d<float> Fluid::get_horizontal_edge_velocity(
    int i,
    int j) const {
  auto v = this->d_vel_y[indx(i, j)];

  float avg_u = this->d_vel_x[indx(i, j)];
  int count = 1;

  if (is_valid_fluid(i + 1, j)) {
    avg_u += this->d_vel_x[indx(i + 1, j)];
    count++;
  }

  if (is_valid_fluid(i, j - 1)) {
    avg_u += this->d_vel_x[indx(i, j - 1)];
    count++;
  }

  if (is_valid_fluid(i + 1, j - 1)) {
    avg_u += this->d_vel_x[indx(i + 1, j - 1)];
    count++;
  }

  avg_u /= count;

  return Vector2d<float>(avg_u, v);
}

__device__ inline float Fluid::get_general_velocity_y(float x, float y) const {
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
      avg_v += w_y * w_x * this->d_vel_y[indx(i, j)];
    }

    if (this->is_valid_fluid(i - 1, j)) {
      avg_v += w_y * (1 - w_x) * this->d_vel_y[indx(i - 1, j)];
    }

    if (this->is_valid_fluid(i - 1, j + 1)) {
      avg_v += (1 - w_y) * (1 - w_x) * this->d_vel_y[indx(i - 1, j + 1)];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_v += (1 - w_y) * w_x * this->d_vel_y[indx(i, j + 1)];
    }
  }
  // take average with the right cell
  else {
    float d_x = in_x - this->cell_size / 2.0;
    float w_x = 1 - d_x / this->cell_size;
    float w_y = 1 - in_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_v += w_y * w_x * this->d_vel_y[indx(i, j)];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_v += (1 - w_y) * w_x * this->d_vel_y[indx(i, j + 1)];
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      avg_v += (1 - w_y) * (1 - w_x) * this->d_vel_y[indx(i + 1, j + 1)];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_v += w_y * (1 - w_x) * this->d_vel_y[indx(i + 1, j)];
    }
  }

  return avg_v;
}

__device__ inline float Fluid::get_general_velocity_x(float x, float y) const {
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
      avg_u += w_y * w_x * this->d_vel_x[indx(i, j)];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_u += w_y * (1 - w_x) * this->d_vel_x[indx(i + 1, j)];
    }

    if (this->is_valid_fluid(i, j - 1)) {
      avg_u += (1 - w_y) * w_x * this->d_vel_x[indx(i, j - 1)];
    }

    if (this->is_valid_fluid(i + 1, j - 1)) {
      avg_u += (1 - w_y) * (1 - w_x) * this->d_vel_x[indx(i + 1, j - 1)];
    }
  }

  // take average with the top cell
  else {
    float d_y = in_y - this->cell_size / 2.0;
    float w_x = 1 - in_x / this->cell_size;
    float w_y = 1 - d_y / this->cell_size;

    if (this->is_valid_fluid(i, j)) {
      avg_u += w_y * w_x * this->d_vel_x[indx(i, j)];
    }

    if (this->is_valid_fluid(i, j + 1)) {
      avg_u += (1 - w_y) * w_x * this->d_vel_x[indx(i, j + 1)];
    }

    if (this->is_valid_fluid(i + 1, j)) {
      avg_u += w_y * (1 - w_x) * this->d_vel_x[indx(i + 1, j)];
    }

    if (this->is_valid_fluid(i + 1, j + 1)) {
      avg_u += (1 - w_y) * (1 - w_x) * this->d_vel_x[indx(i + 1, j + 1)];
    }
  }

  return avg_u;
}

__device__ inline Vector2d<float> Fluid::get_general_velocity(float x,
                                                              float y) const {
  float u = this->get_general_velocity_x(x, y);
  float v = this->get_general_velocity_y(x, y);
  return Vector2d<float>(u, v);
}

__device__ inline Vector2d<float> Fluid::get_center_position(int i,
                                                             int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size,
                         (j + 0.5) * this->cell_size);
}

__device__ inline Vector2d<float> Fluid::get_u_position(int i, int j) const {
  return Vector2d<float>(i * this->cell_size, (j + 0.5) * this->cell_size);
}

__device__ inline Vector2d<float> Fluid::get_v_position(int i, int j) const {
  return Vector2d<float>((i + 0.5) * this->cell_size, j * this->cell_size);
}

__device__ inline void Fluid::apply_smoke_advection_at(int i,
                                                       int j,
                                                       float d_t) {
  Vector2d<float> current_pos = this->get_center_position(i, j);
  Vector2d<float> current_velocity =
      this->get_general_velocity(current_pos.get_x(), current_pos.get_y());
  auto prev_pos = current_pos - current_velocity * d_t;
  float new_smoke = interpolate_smoke(prev_pos.get_x(), prev_pos.get_y());
  this->d_smoke_buffer[indx(i, j)] = new_smoke;
}

__device__ inline void Fluid::update_smoke_advection_at(int i,
                                                        int j,
                                                        float d_t) {
  this->d_smoke[indx(i, j)] = this->d_smoke_buffer[indx(i, j)];
}

__global__ void apply_smoke_advection_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->apply_smoke_advection_at(i, j, d_t);
}

__global__ void update_smoke_advection_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->update_smoke_advection_at(i, j, d_t);
}

inline void Fluid::apply_smoke_advection(float d_t) {
  apply_smoke_advection_kernel<<<this->kernel_grid_dim,
                                 this->kernel_block_dim>>>(d_this, d_t);
  update_smoke_advection_kernel<<<this->kernel_grid_dim,
                                  this->kernel_block_dim>>>(d_this, d_t);
}

__device__ inline void Fluid::apply_velocity_advection_at(int i,
                                                          int j,
                                                          float d_t) {
  Vector2d<float> current_pos = this->get_u_position(i, j);
  Vector2d<float> current_velocity = this->get_vertical_edge_velocity(i, j);
  auto prev_pos = current_pos - current_velocity * d_t;
  float new_velocity =
      this->get_general_velocity_x(prev_pos.get_x(), prev_pos.get_y());
  this->d_vel_x_buffer[indx(i, j)] = new_velocity;

  current_pos = this->get_v_position(i, j);
  current_velocity = this->get_horizontal_edge_velocity(i, j);
  prev_pos = current_pos - current_velocity * d_t;
  new_velocity =
      this->get_general_velocity_y(prev_pos.get_x(), prev_pos.get_y());
  this->d_vel_y_buffer[indx(i, j)] = new_velocity;
}

__device__ inline void Fluid::update_velocity_advection_at(int i,
                                                           int j,
                                                           float d_t) {
  this->d_vel_x[indx(i, j)] = this->d_vel_x_buffer[indx(i, j)];
  this->d_vel_y[indx(i, j)] = this->d_vel_y_buffer[indx(i, j)];
}

__global__ void apply_velocity_advection_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->apply_velocity_advection_at(i, j, d_t);
}

__global__ void update_velocity_advection_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->update_velocity_advection_at(i, j, d_t);
}

inline void Fluid::apply_velocity_advection(float d_t) {
  apply_velocity_advection_kernel<<<this->kernel_grid_dim,
                                    this->kernel_block_dim>>>(d_this, d_t);
  update_velocity_advection_kernel<<<this->kernel_grid_dim,
                                     this->kernel_block_dim>>>(d_this, d_t);
}

__device__ inline float Fluid::interpolate_smoke(float x, float y) const {
  int i = x / this->cell_size;
  int j = y / this->cell_size;

  float in_x = x - i * this->cell_size;
  float in_y = y - j * this->cell_size;

  Vector2d<int> indices_1(i, j);
  Vector2d<int> indices_2;
  Vector2d<int> indices_3;
  Vector2d<int> indices_4;

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
    avg_smoke += w1 * this->d_smoke[indx(indices_1.get_x(), indices_1.get_y())];
  }
  if (is_valid_fluid(indices_2.get_x(), indices_2.get_y())) {
    avg_smoke += w2 * this->d_smoke[indx(indices_2.get_x(), indices_2.get_y())];
  }
  if (is_valid_fluid(indices_3.get_x(), indices_3.get_y())) {
    avg_smoke += w3 * this->d_smoke[indx(indices_3.get_x(), indices_3.get_y())];
  }
  if (is_valid_fluid(indices_4.get_x(), indices_4.get_y())) {
    avg_smoke += w4 * this->d_smoke[indx(indices_4.get_x(), indices_4.get_y())];
  }

  return avg_smoke;
}

// ? review this
// ? review logic
__device__ inline void Fluid::apply_extrapolation_at(int i, int j) {
  if (j == 0) {
    this->d_vel_x[indx(i, j)] = this->d_vel_x[indx(i, j + 1)];
    this->d_vel_y[indx(i, j + 1)] = 0;
  } else if (j == this->height - 1) {
    this->d_vel_x[indx(i, j)] = this->d_vel_x[indx(i, j - 1)];
  }
  if (i == 0) {
    this->d_vel_y[indx(i, j)] = this->d_vel_y[indx(i + 1, j)];
    this->d_vel_x[indx(i + 1, j)] = 0;
  } else if (i == this->width - 1) {
    this->d_vel_y[indx(i, j)] = this->d_vel_y[indx(i - 1, j)];
  }
}

__global__ void apply_extrapolation_kernel(Fluid* d_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->apply_extrapolation_at(i, j);
}

inline void Fluid::apply_extrapolation() {
  apply_extrapolation_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(
      d_this);
}

__global__ void decay_smoke_kernel(Fluid* d_fluid, float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_fluid->decay_smoke_at(i, j, d_t);
}

__device__ inline void Fluid::decay_smoke_at(int i, int j, float d_t) {
#if ENABLE_SMOKE_DECAY
  this->smoke[i][j] = max(smoke - SMOKE_DECAY_RATE * d_t, 0.0);
#endif
}

inline void Fluid::decay_smoke(float d_t) {
  decay_smoke_kernel<<<this->kernel_grid_dim, this->kernel_block_dim>>>(d_this,
                                                                        d_t);
}

// ? put the whole thing into a graph
inline void Fluid::update(float d_t) {
  this->apply_external_forces(d_t);
#if ENABLE_PRESSURE
  this->zero_pressure();
#endif
  this->apply_projection(d_t);
#if ENABLE_PRESSURE
  thrust::device_ptr<float> device_pressure =
      thrust::device_pointer_cast(this->d_pressure);
  this->min_pressure = thrust::reduce(
      device_pressure, device_pressure + (this->width * this->height),
      std::numeric_limits<float>::infinity(), thrust::minimum<float>());
  this->max_pressure = thrust::reduce(
      device_pressure, device_pressure + (this->width * this->height),
      -std::numeric_limits<float>::infinity(), thrust::maximum<float>());
#endif
  this->apply_extrapolation();
  this->apply_velocity_advection(d_t);
#if ENABLE_SMOKE
  if (WIND_SMOKE != 0) {
    this->apply_smoke_advection(d_t);
    this->decay_smoke(d_t);
  }
#endif
  cudaDeviceSynchronize();
}