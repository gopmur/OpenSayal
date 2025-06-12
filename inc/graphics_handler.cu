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

class GraphicsHandler {
 private:
  const int arrow_data_height;
  const int arrow_data_width;
  const int traces_height;
  const int traces_width;

  const float arrow_head_length;
  const float arrow_head_angle;
  const float arrow_disable_thresh_hold;

  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* fluid_texture;
  SDL_PixelFormat* format;
  // SDL_Point traces[W / TRACE_SPACER][H / TRACE_SPACER][TRACE_LENGTH];
  // ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];

  SDL_Point* traces;
  SDL_Point* d_traces;
  ArrowData* arrow_data;
  ArrowData* d_arrow_data;

  GraphicsHandler* d_this;

  inline void draw_arrow(const ArrowData& arrow_data);
  __device__ inline ArrowData make_arrow_data(int x,
                                              int y,
                                              float length,
                                              float angle);
  inline void update_fluid_pixels(const Fluid& fluid);

  inline void update_velocity_arrows(const Fluid& fluid);
  inline void update_center_velocity_arrow(const Fluid& fluid);
  inline void update_traces(const Fluid& fluid, float d_t);

  inline void alloc_device_memory();
  inline void alloc_host_memory();
  inline void init_device_memory();
  inline void free_device_memory();
  inline void free_host_memory();
  inline void init_sdl();

  void cleanup();

 public:
  const int width;
  const int height;
  const int cell_size;

  int* d_fluid_pixels;
  int* fluid_pixels;

  // int fluid_pixels[H][W];

  __device__ __host__ inline int indx(int x, int y);
  __device__ __host__ inline int indx_traces(int i, int j, int n);
  __device__ __host__ inline int indx_arrow_data(int i, int j);

  __device__ inline void update_trace_at(const Fluid* fluid,
                                         float d_t,
                                         int i,
                                         int j);
  __device__ inline void update_center_velocity_arrow_at(const Fluid* fluid,
                                                         int i,
                                                         int j);
  __device__ inline void update_smoke_and_pressure(float smoke,
                                                   float pressure,
                                                   int x,
                                                   int y,
                                                   float min_pressure,
                                                   float max_pressure);
  __device__ inline void update_smoke_pixels(float smoke, int x, int y);
  __device__ inline void update_pressure_pixel(float pressure,
                                               int x,
                                               int y,
                                               float min_pressure,
                                               float max_pressure);
  GraphicsHandler(int width,
                  int height,
                  int cell_size,
                  float arrow_head_length,
                  float arrow_head_angle,
                  float arrow_disable_thresh_hold);
  ~GraphicsHandler();
  void update(const Fluid& fluid, float d_t);
};

__device__ __host__ inline int GraphicsHandler::indx(int x, int y) {
  return y * this->width + x;
}
__device__ __host__ inline int GraphicsHandler::indx_traces(int i,
                                                            int j,
                                                            int k) {
  return (this->height / TRACE_SPACER - j - 1) * this->width * TRACE_LENGTH +
         i * TRACE_LENGTH + k;
}
__device__ __host__ inline int GraphicsHandler::indx_arrow_data(int i, int j) {
  return (this->height / ARROW_SPACER - j - 1) * this->width + i;
}

inline void GraphicsHandler::alloc_device_memory() {
  cudaMalloc(&d_this, sizeof(GraphicsHandler));
  cudaMalloc(
      &this->d_arrow_data,
      this->arrow_data_height * this->arrow_data_width * sizeof(ArrowData));
  cudaMalloc(&this->d_traces,
             this->traces_width * this->traces_height * sizeof(SDL_Point));
  cudaMalloc(&this->d_fluid_pixels, this->width * this->height * sizeof(int));
}

inline void GraphicsHandler::init_device_memory() {
  cudaMemcpy(d_this, this, sizeof(GraphicsHandler), cudaMemcpyHostToDevice);
}

inline void GraphicsHandler::alloc_host_memory() {
  this->fluid_pixels =
      static_cast<int*>(std::malloc(this->height * this->width * sizeof(int)));
  this->arrow_data = static_cast<ArrowData*>(std::malloc(
      this->arrow_data_height * this->arrow_data_width * sizeof(ArrowData)));
  this->traces = static_cast<SDL_Point*>(std::malloc(
      this->traces_height * this->traces_width * sizeof(SDL_Point)));
}

inline void GraphicsHandler::init_sdl() {
  this->window = nullptr;
  this->renderer = nullptr;
  this->fluid_texture = nullptr;
  this->format = nullptr;

  int window_height = cell_size * height;
  int window_width = cell_size * width;

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
                        SDL_TEXTUREACCESS_STREAMING, this->width, this->height);
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
}

GraphicsHandler::GraphicsHandler(int width,
                                 int height,
                                 int cell_size,
                                 float arrow_head_length,
                                 float arrow_head_angle,
                                 float arrow_disable_thresh_hold)
    : width(width),
      height(height),
      cell_size(cell_size),
      arrow_head_angle(arrow_head_angle),
      arrow_head_length(arrow_head_length),
      arrow_disable_thresh_hold(arrow_disable_thresh_hold),
      arrow_data_width(width / ARROW_SPACER),
      arrow_data_height(height / ARROW_SPACER),
      traces_width(width / TRACE_SPACER),
      traces_height(height / TRACE_SPACER) {
  this->alloc_host_memory();
  this->alloc_device_memory();
  this->init_device_memory();
  this->init_sdl();
  Logger::static_debug("graphics initialized successfully");
}

GraphicsHandler::~GraphicsHandler() {
  this->cleanup();
}

inline void GraphicsHandler::free_host_memory() {
  std::free(this->traces);
  std::free(this->arrow_data);
  std::free(this->fluid_pixels);
}

inline void GraphicsHandler::free_device_memory() {
  cudaFree(d_this);
  cudaFree(d_traces);
  cudaFree(d_arrow_data);
  cudaFree(d_fluid_pixels);
}

void GraphicsHandler::cleanup() {
  this->free_device_memory();
  this->free_host_memory();
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

__device__ inline ArrowData GraphicsHandler::make_arrow_data(int x,
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

inline void GraphicsHandler::draw_arrow(const ArrowData& arrow_data) {
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

__device__ inline void GraphicsHandler::update_smoke_pixels(float smoke,
                                                            int x,
                                                            int y) {
  uint8_t color = 255 - static_cast<uint8_t>(smoke * 255);
  this->d_fluid_pixels[indx(x, y)] = map_rgba(255, color, color, 255);
}

__device__ inline void GraphicsHandler::update_smoke_and_pressure(
    float smoke,
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
  this->d_fluid_pixels[indx(x, y)] = map_rgba(r, g, b, 255);
}

__device__ inline void GraphicsHandler::update_pressure_pixel(
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
  this->d_fluid_pixels[indx(x, y)] = map_rgba(r, g, b, 255);
}

__global__ void update_smoke_and_pressure_pixel_kernel(
    Fluid* d_fluid,
    GraphicsHandler* d_graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_smoke_and_pressure(
        d_fluid->d_smoke[d_fluid->indx(i, j)],
        d_fluid->d_pressure[d_fluid->indx(i, j)], x, y, min_pressure,
        max_pressure);
  }
}

__global__ void update_smoke_pixel_kernel(Fluid* d_fluid,
                                          GraphicsHandler* d_graphics_handler) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_smoke_pixels(
        d_fluid->d_smoke[d_fluid->indx(i, j)], x, y);
  }
}

__global__ void update_pressure_pixel_kernel(
    Fluid* d_fluid,
    GraphicsHandler* d_graphics_handler,
    float min_pressure,
    float max_pressure) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  int x = i;
  int y = d_fluid->height - j - 1;
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    d_graphics_handler->d_fluid_pixels[d_graphics_handler->indx(x, y)] =
        map_rgba(80, 80, 80, 255);
  } else {
    d_graphics_handler->update_pressure_pixel(
        d_fluid->d_pressure[d_fluid->indx(i, j)], x, y, min_pressure,
        max_pressure);
  }
}

inline void GraphicsHandler::update_fluid_pixels(const Fluid& fluid) {
  float min_pressure = fluid.min_pressure;
  float max_pressure = fluid.max_pressure;
  int block_dim_x = BLOCK_SIZE_X;
  int block_dim_y = BLOCK_SIZE_Y;
  int grid_dim_x = std::ceil(static_cast<float>(fluid.width) / block_dim_x);
  int grid_dim_y = std::ceil(static_cast<float>(fluid.height) / block_dim_y);
  auto block_dim = dim3(block_dim_x, block_dim_y, 1);
  auto grid_dim = dim3(grid_dim_x, grid_dim_y, 1);
#if ENABLE_PRESSURE and ENABLE_SMOKE
  update_smoke_and_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
      fluid.d_this, d_this, min_pressure, max_pressure);
#elif ENABLE_PRESSURE
  update_pressure_pixel_kernel<<<grid_dim, block_dim>>>(
      fluid.d_this, d_this, min_pressure, max_pressure);
#elif ENABLE_SMOKE
  update_smoke_pixel_kernel<<<grid_dim, block_dim>>>(fluid.d_this, d_this);
#endif

  cudaMemcpyAsync(this->fluid_pixels, this->d_fluid_pixels,
                  sizeof(int) * this->height * this->width,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

__global__ void update_center_velocity_arrow_kernel(
    GraphicsHandler* d_graphics_handler,
    Fluid* d_fluid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= ARROW_SPACER;
  j *= ARROW_SPACER;
  if (i >= d_fluid->width or j >= d_fluid->height) {
    return;
  }
  d_graphics_handler->update_center_velocity_arrow_at(d_fluid, i, j);
}

__device__ void GraphicsHandler::update_center_velocity_arrow_at(
    const Fluid* d_fluid,
    int i,
    int j) {
  if (d_fluid->d_is_solid[d_fluid->indx(i, j)]) {
    return;
  }
  float x = (i + 0.5) * this->cell_size;
  float y = (this->height - j - 1 + 0.5) * this->cell_size;
  Vector2d<float> velocity =
      d_fluid->get_general_velocity(x, this->height * this->cell_size - y);
  auto vel_x = velocity.get_x();
  auto vel_y = velocity.get_y();
  auto angle = atan2(vel_y, vel_x);
  auto length = sqrt(vel_x * vel_x + vel_y * vel_y);
  arrow_data[this->indx_arrow_data(i / ARROW_SPACER, j / ARROW_SPACER)] =
      this->make_arrow_data(x, y, length, angle);
}

inline void GraphicsHandler::update_center_velocity_arrow(const Fluid& fluid) {
  int arrow_x = this->width / ARROW_SPACER;
  int arrow_y = this->height / ARROW_SPACER;
  int block_x = BLOCK_SIZE_X;
  int block_y = BLOCK_SIZE_Y;
  int grid_x = std::ceil(arrow_x / static_cast<float>(BLOCK_SIZE_X));
  int grid_y = std::ceil(arrow_y / static_cast<float>(BLOCK_SIZE_Y));
  auto block_dim = dim3(block_x, block_y, 1);
  auto gird_dim = dim3(grid_x, grid_y, 1);
  update_center_velocity_arrow_kernel<<<gird_dim, block_dim>>>(d_this,
                                                               fluid.d_this);
  cudaMemcpyAsync(this->arrow_data, this->d_arrow_data,
                  sizeof(ArrowData) * arrow_x * arrow_y,
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int j = 0; j < this->height / ARROW_SPACER; j++) {
    for (int i = 0; i < this->width / ARROW_SPACER; i++) {
      this->draw_arrow(arrow_data[indx_arrow_data(i, j)]);
    }
  }
}

inline void GraphicsHandler::update_velocity_arrows(const Fluid& fluid) {
#if DRAW_CENTER_ARROW
  SDL_SetRenderDrawColor(renderer, CENTER_ARROW_COLOR);
  this->update_center_velocity_arrow(fluid);
#endif
}

__device__ inline void GraphicsHandler::update_trace_at(const Fluid* fluid,
                                                        float d_t,
                                                        int i,
                                                        int j) {
  const int trace_i = i / TRACE_SPACER;
  const int trace_j = j / TRACE_SPACER;

  if (fluid->d_is_solid[fluid->indx(i, j)]) {
    this->traces[indx_traces(trace_i, trace_j, 0)] = {-1, -1};
  } else {
    fluid->trace(i, j, d_t, &this->traces[indx_traces(trace_i, trace_j, 0)]);
  }
}

__global__ void update_traces_kernel(GraphicsHandler* d_graphics_handler,
                                     Fluid* d_fluid,
                                     float d_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  i *= TRACE_SPACER;
  j *= TRACE_SPACER;

  if (i >= d_fluid->width || j >= d_fluid->height)
    return;

  d_graphics_handler->update_trace_at(d_fluid, d_t, i, j);
}

inline void GraphicsHandler::update_traces(const Fluid& fluid, float d_t) {
  const int trace_cols = this->width / TRACE_SPACER;
  const int trace_rows = this->height / TRACE_SPACER;

  dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 grid_dim((trace_cols + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                (trace_rows + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

  update_traces_kernel<<<grid_dim, block_dim>>>(d_this, fluid.d_this, d_t);
  cudaMemcpyAsync(this->traces, this->d_traces,
                  trace_cols * trace_rows * TRACE_LENGTH * sizeof(SDL_Point),
                  cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  SDL_SetRenderDrawColor(this->renderer, TRACE_COLOR);
  for (int j = 0; j < trace_rows; j++) {
    for (int i = 0; i < trace_cols; i++) {
      if (traces[indx_traces(i, j, 0)].x < 0)
        continue;
      SDL_RenderDrawLines(renderer, &traces[indx_traces(i, j, 0)],
                          TRACE_LENGTH);
    }
  }
}

void GraphicsHandler::update(const Fluid& fluid, float d_t) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels,
                    this->width * sizeof(int));

  // Render the texture
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, this->fluid_texture, NULL, NULL);
#if ENABLE_TRACES
  this->update_traces(fluid, d_t);
#endif
  this->update_velocity_arrows(fluid);

  SDL_RenderPresent(renderer);
}