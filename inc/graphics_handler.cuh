#pragma once

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <format>

#include "SDL.h"
#include "SDL_rect.h"
#include "SDL_render.h"

#include "config.hpp"
#include "fluid.cuh"
#include "helper.cuh"
#include "logger.hpp"

struct ArrowData {
  int start_x;
  int start_y;
  int end_x;
  int end_y;
  int right_head_end_x;
  int right_head_end_y;
  int left_head_end_x;
  int left_head_end_y;
  bool valid;
};

class GraphicsHandler {
 private:
  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* fluid_texture;
  SDL_PixelFormat* format;

  int* traces_x;
  int* d_traces_x;
  int* traces_y;
  int* d_traces_y;
  ArrowData* arrow_data;
  ArrowData* d_arrow_data;

  GraphicsHandler* d_this;

  void draw_arrow(const ArrowData& arrow_data);
  __device__ ArrowData make_arrow_data(int x, int y, float length, float angle);
  void update_fluid_pixels(const Fluid& fluid);

  void update_velocity_arrows(const Fluid& fluid);
  void update_center_velocity_arrow(const Fluid& fluid);
  void update_traces(const Fluid& fluid, float d_t);

  void alloc_device_memory();
  void alloc_host_memory();
  void init_device_memory();
  void free_device_memory();
  void free_host_memory();
  void init_sdl();

  void cleanup();

 public:
  const int width;
  const int height;
  const int cell_size;

  const int block_size_x;
  const int block_size_y;

  const int arrow_data_height;
  const int arrow_data_width;
  const int traces_height;
  const int traces_width;

  const float arrow_head_length;
  const float arrow_head_angle;
  const float arrow_disable_thresh_hold;
  const float arrow_length_multiplier;
  const int arrow_distance;

  const int trace_distance;
  const int trace_length;

  const ColorConfig arrow_color;
  const ColorConfig trace_color;

  const bool enable_pressure;
  const bool enable_smoke;

  const bool enable_traces;
  const bool enable_arrows;

  int* d_fluid_pixels;
  int* fluid_pixels;

  __device__ __host__ int indx(int x, int y);
  __device__ __host__ int indx_traces(int i, int j, int n);
  __device__ __host__ int indx_arrow_data(int i, int j);

  __device__ void update_trace_at(const Fluid* fluid, float d_t, int i, int j);
  __device__ void update_center_velocity_arrow_at(const Fluid* fluid,
                                                  int i,
                                                  int j);
  __device__ void update_smoke_and_pressure(float smoke,
                                            float pressure,
                                            int x,
                                            int y,
                                            float min_pressure,
                                            float max_pressure);
  __device__ void update_smoke_pixels(float smoke, int x, int y);
  __device__ void update_pressure_pixel(float pressure,
                                        int x,
                                        int y,
                                        float min_pressure,
                                        float max_pressure);
  GraphicsHandler(Config config);
  ~GraphicsHandler();
  void update(const Fluid& fluid, float d_t);
};
