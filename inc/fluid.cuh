#pragma once

#include "SDL_rect.h"

#include "config_parser.hpp"
#include "helper.cuh"

struct Source {
  bool active;
  float smoke;
  float velocity;
  Vector2d<int> position;
};

class Fluid {
 private:
  __device__ float interpolate_smoke(float x, float y) const;
  __device__ float get_general_velocity_y(float x, float y) const;
  __device__ float get_general_velocity_x(float x, float y) const;
  __device__ __host__ bool index_is_valid(int i, int j) const;
  __device__ bool is_valid_fluid(int i, int j) const;
  __device__ Vector2d<float> get_center_position(int i, int j) const;
  __device__ Vector2d<float> get_u_position(int i, int j) const;
  __device__ Vector2d<float> get_v_position(int, int j) const;

  void zero_pressure();

  void apply_external_forces(Source source, float d_t);
  void apply_projection(float d_t);
  void apply_smoke_advection(float d_t);
  void apply_velocity_advection(float d_t);
  void apply_extrapolation();
  void apply_diffusion(float d_t);
  void decay_smoke(float d_t);

  void alloc_device_memory();
  void init_device_memory(Config config);

 public:
  const int width;
  const int height;
  const float g;
  const float density;
  const float viscosity;
  const float o;
  const int cell_size;
  const int n;
  const float drag_coeff;
  const int wind_tunnel_smoke_length;
  const int wind_tunnel_smoke_count;
  const int wind_tunnel_smoke_height;
  const float wind_tunnel_speed;
  const int wind_tunnel_height;
  const float wind_tunnel_smoke;
  const bool enable_smoke_decay;
  const bool enable_pressure;
  const bool enable_smoke;
  const bool enable_interactive;
  const float smoke_decay_rate;

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

  Fluid(Config config);
  ~Fluid();

  __host__ __device__ int indx(int i, int j) const;

  __device__ float get_divergence(int i, int j) const;

  __device__ __host__ bool is_edge(int i, int j) const;

  __device__ Vector2d<float> get_general_velocity(float x, float y) const;
  __device__ Vector2d<float> get_vertical_edge_velocity(int i, int j) const;
  __device__ Vector2d<float> get_horizontal_edge_velocity(int i, int j) const;
  __device__ void trace(int i,
                        int j,
                        float d_t,
                        int* trace_line_x,
                        int* trace_line_y,
                        int trace_length) const;

  __device__ void apply_external_forces_at(Source source,
                                           int i,
                                           int j,
                                           float d_t);
  __device__ void apply_projection_at(int i, int j, float d_t);
  __device__ void apply_velocity_advection_at(int i, int j, float d_t);
  __device__ void update_pressure_at(int i,
                                     int j,
                                     float velocity_diff,
                                     float d_t);
  __device__ void apply_smoke_advection_at(int i, int j, float d_t);
  __device__ void update_smoke_advection_at(int i, int j, float d_t);
  __device__ void update_velocity_advection_at(int i, int j, float d_t);
  __device__ void apply_extrapolation_at(int i, int j);
  __device__ void decay_smoke_at(int i, int j, float d_t);
  __device__ void zero_pressure_at(int i, int j);
  __device__ void apply_diffusion_at(int i, int j, float d_t);

  void update(Source source, float d_t);
};