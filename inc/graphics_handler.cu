#pragma once

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <format>

#include "SDL.h"
#include "SDL_rect.h"
#include "SDL_render.h"

#include "config.hpp"
#include "fluid.cu"
#include "helper.cu"
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

template <int H, int W, int S>
class GraphicsHandler {
 private:
  float arrow_head_length;
  float arrow_head_angle;
  float arrow_disable_thresh_hold;
  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* fluid_texture;
  SDL_PixelFormat* format;
  SDL_Point traces[W / TRACE_SPACER][H / TRACE_SPACER][TRACE_LENGTH];
  ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];
  GraphicsHandler<H, W, S>* device_graphics_handler;

  inline void draw_arrow(const ArrowData& arrow_data);
  __device__ __host__ inline ArrowData make_arrow_data(int x,
                                                       int y,
                                                       float length,
                                                       float angle);
  inline void update_fluid_pixels(const Fluid& fluid);

  inline void update_velocity_arrows(const Fluid& fluid);
  inline void update_center_velocity_arrow(const Fluid& fluid);
  inline void update_traces(const Fluid& fluid, float d_t);

  void cleanup();

 public:
  int fluid_pixels[H][W];
  __device__ inline void update_trace_at(const Fluid* fluid,
                                         float d_t,
                                         int i,
                                         int j);
  __device__ inline void update_center_velocity_arrow_at(const Fluid* fluid,
                                                         int i,
                                                         int j);
  __host__ __device__ inline void update_smoke_and_pressure(float smoke,
                                                            float pressure,
                                                            int x,
                                                            int y,
                                                            float min_pressure,
                                                            float max_pressure);
  __host__ __device__ inline void update_smoke_pixels(float smoke,
                                                      int x,
                                                      int y);
  __host__ __device__ inline void update_pressure_pixel(float pressure,
                                                        int x,
                                                        int y,
                                                        float min_pressure,
                                                        float max_pressure);
  GraphicsHandler(float arrow_head_length,
                  float arrow_head_angle,
                  float arrow_disable_thresh_hold);
  ~GraphicsHandler();
  void update(const Fluid& fluid, float d_t);
};

template <int H, int W, int S>
GraphicsHandler<H, W, S>::GraphicsHandler(float arrow_head_length,
                                          float arrow_head_angle,
                                          float arrow_disable_thresh_hold)
    : arrow_head_angle(arrow_head_angle),
      arrow_head_length(arrow_head_length),
      arrow_disable_thresh_hold(arrow_disable_thresh_hold) {
  cudaMalloc(&this->device_graphics_handler, sizeof(GraphicsHandler<H, W, S>));

  this->window = nullptr;
  this->renderer = nullptr;
  this->fluid_texture = nullptr;
  this->format = nullptr;

  int window_height = S * H;
  int window_width = S * W;

  auto sdl_status = SDL_Init(SDL_INIT_VIDEO);
  if (sdl_status < 0) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(
        std::format("video initialization failed: ", sdl_error_message));
    exit(EXIT_FAILURE);
  }

  this->window = SDL_CreateWindow("Fluid simulation", SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED, window_width,
                                  window_height, SDL_WINDOW_SHOWN);
  if (this->window == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("window creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->renderer =
      SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
  if (this->renderer == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("renderer creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->fluid_texture =
      SDL_CreateTexture(this->renderer, SDL_PIXELFORMAT_RGBA8888,
                        SDL_TEXTUREACCESS_STREAMING, W, H);
  if (this->fluid_texture == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("texture creation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
  if (this->format == nullptr) {
    auto sdl_error_message = SDL_GetError();
    Logger::error(std::format("format allocation failed: ", sdl_error_message));
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(this->device_graphics_handler, this,
             sizeof(GraphicsHandler<H, W, S>), cudaMemcpyHostToDevice);
  Logger::static_debug("graphics initialized successfully");
}

template <int H, int W, int S>
GraphicsHandler<H, W, S>::~GraphicsHandler() {
  this->cleanup();
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::cleanup() {
  cudaFree(this->device_graphics_handler);
  Logger::static_debug("cleaning up graphics");
  if (this->window != nullptr) {
    SDL_DestroyWindow(this->window);
  }
  if (this->renderer != nullptr) {
    SDL_DestroyRenderer(this->renderer);
  }
  if (this->fluid_texture != nullptr) {
    SDL_DestroyTexture(this->fluid_texture);
  }
  if (this->format != nullptr) {
    SDL_FreeFormat(format);
  }
  SDL_Quit();
}

template <int H, int W, int S>
__device__ __host__ inline ArrowData GraphicsHandler<H, W, S>::make_arrow_data(
    int x,
    int y,
    float length,
    float angle) {
  ArrowData arrow_data;

  if (length < ARROW_DISABLE_THRESH_HOLD) {
    arrow_data.valid = false;
    return arrow_data;
  }

  arrow_data.valid = true;

  arrow_data.start_x = x;
  arrow_data.start_y = y;
  length *= ARROW_LENGTH_MULTIPLIER;
  int x_offset = length * cos(angle);
  int y_offset = -length * sin(angle);
  arrow_data.end_x = x + x_offset;
  arrow_data.end_y = y + y_offset;

  int arrow_x_offset = -arrow_head_length * cos(angle + arrow_head_angle);
  int arrow_y_offset = arrow_head_length * sin(angle + arrow_head_angle);
  arrow_data.left_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.left_head_end_y = arrow_data.end_y + arrow_y_offset;

  arrow_x_offset = -arrow_head_length * cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * sin(arrow_head_angle - angle);
  arrow_data.right_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.right_head_end_y = arrow_data.end_y + arrow_y_offset;

  return arrow_data;
};

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::draw_arrow(const ArrowData& arrow_data) {
  if (not arrow_data.valid) {
    return;
  }
  SDL_RenderDrawLine(renderer, arrow_data.start_x, arrow_data.start_y,
                     arrow_data.end_x, arrow_data.end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.left_head_end_x, arrow_data.left_head_end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.right_head_end_x, arrow_data.right_head_end_y);
}

template <int H, int W, int S>
__host__ __device__ inline void
GraphicsHandler<H, W, S>::update_smoke_pixels(float smoke, int x, int y) {
  uint8_t color = 255 - static_cast<uint8_t>(smoke * 255);
  this->fluid_pixels[y][x] = map_rgba(255, color, color, 255);
}

template <int H, int W, int S>
__host__ __device__ inline void
GraphicsHandler<H, W, S>::update_smoke_and_pressure(float smoke,
                                                    float pressure,
                                                    int x,
                                                    int y,
                                                    float min_pressure,
                                                    float max_pressure) {
  float norm_p = 0;
  if (pressure < 0 and min_pressure != 0) {
    norm_p = -pressure / min_pressure;
  } else if (max_pressure != 0) {
    norm_p = pressure / max_pressure;
  }
  norm_p = clamp(norm_p, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, smoke, r, g, b);
  this->fluid_pixels[y][x] = map_rgba(r, g, b, 255);
}

template <int H, int W, int S>
__host__ __device__ inline void GraphicsHandler<H, W, S>::update_pressure_pixel(
    float pressure,
    int x,
    int y,
    float min_pressure,
    float max_pressure) {
  float norm_p;
  if (pressure < 0) {
    norm_p = -pressure / min_pressure;
  } else {
    norm_p = pressure / max_pressure;
  }
  norm_p = clamp(norm_p, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, 1.0f, r, g, b);
  this->fluid_pixels[y][x] = map_rgba(r, g, b, 255);
}

template <int H, int W, int S>
__global__ void update_fluid_pixels_kernel(
    GraphicsHandler<H, W, S>* graphics_handler,
    Fluid* fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
}

template <int H, int W, int S>
__global__ void update_smoke_and_pressure_pixel_kernel(
    Fluid* fluid,
    GraphicsHandler<H, W, S>* graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  int x = i;
  int y = H - j - 1;
  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    graphics_handler->fluid_pixels[y][x] = map_rgba(80, 80, 80, 255);
  } else {
    graphics_handler->update_smoke_and_pressure(
        fluid->d_smoke[fluid->indx(i, j)], fluid->d_pressure[fluid->indx(i, j)],
        x, y, min_pressure, max_pressure);
  }
}

template <int H, int W, int S>
__global__ void update_smoke_pixel_kernel(
    Fluid* fluid,
    GraphicsHandler<H, W, S>* graphics_handler) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  int x = i;
  int y = H - j - 1;
  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    graphics_handler->fluid_pixels[y][x] = map_rgba(80, 80, 80, 255);
  } else {
    graphics_handler->update_smoke_pixels(fluid->d_smoke[fluid->indx(i, j)], x,
                                          y);
  }
}

template <int H, int W, int S>
__global__ void update_pressure_pixel_kernel(
    Fluid* fluid,
    GraphicsHandler<H, W, S>* graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W or j >= H) {
    return;
  }
  int x = i;
  int y = H - j - 1;
  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    graphics_handler->fluid_pixels[y][x] = map_rgba(80, 80, 80, 255);
  } else {
    graphics_handler->update_pressure_pixel(
        fluid->d_pressure[fluid->indx(i, j)], x, y, min_pressure, max_pressure);
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_fluid_pixels(const Fluid& fluid) {
  float min_pressure = fluid.min_pressure;
  float max_pressure = fluid.max_pressure;
  int block_dim_x = BLOCK_SIZE_X;
  int block_dim_y = BLOCK_SIZE_Y;
  int grid_dim_x = std::ceil(static_cast<float>(W) / block_dim_x);
  int grid_dim_y = std::ceil(static_cast<float>(H) / block_dim_y);
  auto block_dim = dim3(block_dim_x, block_dim_y, 1);
  auto grid_dim = dim3(grid_dim_x, grid_dim_y, 1);
#if ENABLE_PRESSURE and ENABLE_SMOKE
  update_smoke_and_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
      fluid.d_this, this->device_graphics_handler, min_pressure, max_pressure);
#elif ENABLE_PRESSURE
  update_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
      fluid.device_fluid, this->device_graphics_handler, min_pressure,
      max_pressure);
#elif ENABLE_SMOKE
  update_smoke_pixel_kernel<<<grid_dim, block_dim>>>(
      fluid.device_fluid, this->device_graphics_handler);
#endif

  cudaMemcpyAsync(this->fluid_pixels,
                  this->device_graphics_handler->fluid_pixels,
                  sizeof(int) * H * W, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

template <int H, int W, int S>
__global__ void update_center_velocity_arrow_kernel(
    GraphicsHandler<H, W, S>* graphics_handler,
    Fluid* fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= ARROW_SPACER;
  j *= ARROW_SPACER;
  if (i >= W or j >= H) {
    return;
  }
  graphics_handler->update_center_velocity_arrow_at(fluid, i, j);
}

template <int H, int W, int S>
__device__ void GraphicsHandler<H, W, S>::update_center_velocity_arrow_at(
    const Fluid* fluid,
    int i,
    int j) {
  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    return;
  }
  float x = (i + 0.5) * S;
  float y = (H - j - 1 + 0.5) * S;
  Vector2d<float> velocity = fluid->get_general_velocity(x, H * S - y);
  auto vel_x = velocity.get_x();
  auto vel_y = velocity.get_y();
  auto angle = atan2(vel_y, vel_x);
  auto length = sqrt(vel_x * vel_x + vel_y * vel_y);
  arrow_data[i / ARROW_SPACER][j / ARROW_SPACER] =
      this->make_arrow_data(x, y, length, angle);
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_center_velocity_arrow(
    const Fluid& fluid) {
  int arrow_x = W / ARROW_SPACER;
  int arrow_y = H / ARROW_SPACER;
  int block_x = BLOCK_SIZE_X;
  int block_y = BLOCK_SIZE_Y;
  int grid_x = std::ceil(arrow_x / static_cast<float>(BLOCK_SIZE_X));
  int grid_y = std::ceil(arrow_y / static_cast<float>(BLOCK_SIZE_Y));
  auto block_dim = dim3(block_x, block_y, 1);
  auto gird_dim = dim3(grid_x, grid_y, 1);
  update_center_velocity_arrow_kernel<<<gird_dim, block_dim>>>(
      this->device_graphics_handler, fluid.d_this);
  cudaMemcpyAsync(this->arrow_data, this->device_graphics_handler->arrow_data,
                  sizeof(ArrowData) * arrow_x * arrow_y,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < W / ARROW_SPACER; i++) {
    for (int j = 0; j < H / ARROW_SPACER; j++) {
      this->draw_arrow(arrow_data[i][j]);
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_velocity_arrows(
    const Fluid& fluid) {
#if DRAW_CENTER_ARROW
  SDL_SetRenderDrawColor(renderer, CENTER_ARROW_COLOR);
  this->update_center_velocity_arrow(fluid);
#endif
}

template <int H, int W, int S>
__device__ inline void GraphicsHandler<H, W, S>::update_trace_at(
    const Fluid* fluid,
    float d_t,
    int i,
    int j) {
  const int trace_i = i / TRACE_SPACER;
  const int trace_j = j / TRACE_SPACER;

  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    this->traces[trace_i][trace_j][0] = {-1, -1};
  } else {
    fluid->trace(i, j, d_t, &this->traces[trace_i][trace_j][0]);
  }
}

// Kernel
template <int H, int W, int S>
__global__ void update_traces_kernel(GraphicsHandler<H, W, S>* graphics_handler,
                                     Fluid* fluid,
                                     float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= TRACE_SPACER;
  j *= TRACE_SPACER;

  if (i >= W || j >= H)
    return;

  graphics_handler->update_trace_at(fluid, d_t, i, j);
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_traces(const Fluid& fluid,
                                                    float d_t) {
  const int trace_cols = W / TRACE_SPACER;
  const int trace_rows = H / TRACE_SPACER;

  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid_dim((trace_cols + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (trace_rows + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

  update_traces_kernel<<<grid_dim, block_dim>>>(this->device_graphics_handler,
                                                fluid.d_this, d_t);
  cudaMemcpyAsync(this->traces, this->device_graphics_handler->traces,
                  trace_cols * trace_rows * TRACE_LENGTH * sizeof(SDL_Point),
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  SDL_SetRenderDrawColor(this->renderer, TRACE_COLOR);
  for (int i = 0; i < trace_cols; i++) {
    for (int j = 0; j < trace_rows; j++) {
      if (traces[i][j][0].x < 0)
        continue;
      SDL_RenderDrawLines(renderer, traces[i][j], TRACE_LENGTH);
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update(const Fluid& fluid, float d_t) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels,
                    W * sizeof(int));

  // Render the texture
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, this->fluid_texture, NULL, NULL);
#if ENABLE_TRACES
  this->update_traces(fluid, d_t);
#endif
  this->update_velocity_arrows(fluid);

  SDL_RenderPresent(renderer);
}