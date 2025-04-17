#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <format>

#include "SDL.h"
#include "SDL_rect.h"
#include "SDL_render.h"

#include "config.hpp"
#include "fluid.hpp"
#include "helper.hpp"
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
  std::array<std::array<int, W>, H> fluid_pixels;

  inline void draw_arrow(const ArrowData& arrow_data);
  inline ArrowData make_arrow_data(int x, int y, float length, float angle);
  inline void update_fluid_pixels(const Fluid<H, W>& fluid);
  inline void update_smoke_pixels(const Cell& cell, int x, int y);
  inline void update_pressure_pixels(const Cell& cell,
                                     int x,
                                     int y,
                                     float min_pressure,
                                     float max_pressure);
  inline void update_smoke_and_pressure(const Cell& cell,
                                        int x,
                                        int y,
                                        float min_pressure,
                                        float max_pressure);
  inline void update_velocity_arrows(const Fluid<H, W>& fluid);
  inline void update_center_velocity_arrow(const Fluid<H, W>& fluid);
  inline void update_horizontal_edge_velocity_arrow(const Fluid<H, W>& fluid);
  inline void update_vertical_edge_velocity_arrow(const Fluid<H, W>& fluid);
  inline void update_corner_velocity_arrow(const Fluid<H, W>& fluid);
  inline void update_traces(const Fluid<H, W>& fluid, float d_t);
  void cleanup();

 public:
  GraphicsHandler(float arrow_head_length,
                  float arrow_head_angle,
                  float arrow_disable_thresh_hold);
  ~GraphicsHandler();
  void update(const Fluid<H, W>& fluid, float d_t);
};

template <int H, int W, int S>
GraphicsHandler<H, W, S>::GraphicsHandler(float arrow_head_length,
                                          float arrow_head_angle,
                                          float arrow_disable_thresh_hold)
    : arrow_head_angle(arrow_head_angle),
      arrow_head_length(arrow_head_length),
      arrow_disable_thresh_hold(arrow_disable_thresh_hold) {
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

  Logger::static_debug("graphics initialized successfully");
}

template <int H, int W, int S>
GraphicsHandler<H, W, S>::~GraphicsHandler() {
  this->cleanup();
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::cleanup() {
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
inline ArrowData GraphicsHandler<H, W, S>::make_arrow_data(int x,
                                                           int y,
                                                           float length,
                                                           float angle) {
  ArrowData arrow_data;
  arrow_data.start_x = x;
  arrow_data.start_y = y;
  length *= ARROW_LENGTH_MULTIPLIER;
  int x_offset = length * std::cos(angle);
  int y_offset = -length * std::sin(angle);
  arrow_data.end_x = x + x_offset;
  arrow_data.end_y = y + y_offset;

  int arrow_x_offset = -arrow_head_length * std::cos(angle + arrow_head_angle);
  int arrow_y_offset = arrow_head_length * std::sin(angle + arrow_head_angle);
  arrow_data.left_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.left_head_end_y = arrow_data.end_y + arrow_y_offset;

  arrow_x_offset = -arrow_head_length * std::cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * std::sin(arrow_head_angle - angle);
  arrow_data.right_head_end_x = arrow_data.end_x + arrow_x_offset;
  arrow_data.right_head_end_y = arrow_data.end_y + arrow_y_offset;

  return arrow_data;
};

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::draw_arrow(const ArrowData& arrow_data) {
  SDL_RenderDrawLine(renderer, arrow_data.start_x, arrow_data.start_y,
                     arrow_data.end_x, arrow_data.end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.left_head_end_x, arrow_data.left_head_end_y);
  SDL_RenderDrawLine(renderer, arrow_data.end_x, arrow_data.end_y,
                     arrow_data.right_head_end_x, arrow_data.right_head_end_y);
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_smoke_pixels(const Cell& cell,
                                                          int x,
                                                          int y) {
  auto smoke = cell.get_smoke();
  uint8_t color = 255 - static_cast<uint8_t>(smoke * 255);
  this->fluid_pixels[y][x] = SDL_MapRGBA(this->format, 255, color, color, 255);
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_smoke_and_pressure(
    const Cell& cell,
    int x,
    int y,
    float min_pressure,
    float max_pressure) {
  auto pressure = cell.get_pressure();
  float range = std::max(std::abs(min_pressure), std::abs(max_pressure));
  float norm_p = std::clamp(pressure / range, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  auto smoke = cell.get_smoke();
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, smoke, r, g, b);
  this->fluid_pixels[y][x] = SDL_MapRGBA(this->format, r, g, b, 255);
}
template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_pressure_pixels(
    const Cell& cell,
    int x,
    int y,
    float min_pressure,
    float max_pressure) {
  auto pressure = cell.get_pressure();
  float range = std::max(std::abs(min_pressure), std::abs(max_pressure));
  float norm_p = std::clamp(pressure / range, -1.0f, 1.0f);
  float hue = (1.0f - norm_p) * 120.0f;
  uint8_t r, g, b;
  hsv_to_rgb(hue, 1.0f, 1.0f, r, g, b);
  this->fluid_pixels[y][x] = SDL_MapRGBA(this->format, r, g, b, 255);
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_fluid_pixels(
    const Fluid<H, W>& fluid) {
#if ENABLE_PRESSURE
  float max_pressure = -INFINITY;
  float min_pressure = INFINITY;

  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
      float pressure = fluid.get_pressure(i, j);
      if (pressure > max_pressure) {
        max_pressure = pressure;
      }
      if (pressure < min_pressure) {
        min_pressure = pressure;
      }
    }
  }
#endif

  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      const Cell& cell = fluid.get_cell(i, j);

      int x = i;
      int y = H - j - 1;

      if (cell.is_solid()) {
        this->fluid_pixels[y][x] = SDL_MapRGBA(this->format, 80, 80, 80, 255);
      } else {
#if ENABLE_PRESSURE and ENABLE_SMOKE
        this->update_smoke_and_pressure(cell, x, y, min_pressure, max_pressure);
#elif ENABLE_PRESSURE
        this->update_pressure_pixels(cell, x, y, min_pressure, max_pressure);
#elif ENABLE_SMOKE
        this->update_smoke_pixels(cell, x, y);
#endif
      }
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_center_velocity_arrow(
    const Fluid<H, W>& fluid) {
  ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];
  for (int i = 1; i < W - 1; i += ARROW_SPACER + 1) {
    for (int j = 1; j < H - 1; j += ARROW_SPACER + 1) {
      const Cell& cell = fluid.get_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }
      float x = (i + 0.5) * S;
      float y = (H - j - 1 + 0.5) * S;
      Vector2d<float> velocity = fluid.get_general_velocity(x, H * S - y);
      auto vel_x = velocity.get_x();
      auto vel_y = velocity.get_y();
      auto angle = std::atan2(vel_y, vel_x);
      auto length = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      arrow_data[i / ARROW_SPACER][j / ARROW_SPACER] =
          this->make_arrow_data(x, y, length, angle);
    }
  }
  for (int i = 0; i < W / ARROW_SPACER; i++) {
    for (int j = 0; j < H / ARROW_SPACER; j++) {
      this->draw_arrow(arrow_data[i][j]);
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_horizontal_edge_velocity_arrow(
    const Fluid<H, W>& fluid) {
  ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];
  for (int i = 1; i < W - 1; i += ARROW_SPACER + 1) {
    for (int j = 1; j < H - 1; j += ARROW_SPACER + 1) {
      const Cell& cell = fluid.get_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }
      float x = (i + 0.5) * S;
      float y = (H - j - 1) * S;
      Vector2d<float> velocity = fluid.get_horizontal_edge_velocity(i, j);
      auto vel_x = velocity.get_x();
      auto vel_y = velocity.get_y();
      auto angle = std::atan2(vel_y, vel_x);
      auto length = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      arrow_data[i / ARROW_SPACER][j / ARROW_SPACER] =
          this->make_arrow_data(x, y, length, angle);
    }
  }
  for (int i = 0; i < W / ARROW_SPACER; i++) {
    for (int j = 0; j < H / ARROW_SPACER; j++) {
      this->draw_arrow(arrow_data[i][j]);
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_vertical_edge_velocity_arrow(
    const Fluid<H, W>& fluid) {
  ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];

  for (int i = 1; i < W - 1; i += ARROW_SPACER + 1) {
    for (int j = 1; j < H - 1; j += ARROW_SPACER + 1) {
      const Cell& cell = fluid.get_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }
      float x = (i)*S;
      float y = (H - j - 1 + 0.5) * S;
      Vector2d<float> velocity = fluid.get_vertical_edge_velocity(i, j);
      auto vel_x = velocity.get_x();
      auto vel_y = velocity.get_y();
      auto angle = std::atan2(vel_y, vel_x);
      auto length = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      arrow_data[i / ARROW_SPACER][j / ARROW_SPACER] =
          this->make_arrow_data(x, y, length, angle);
    }
  }
  for (int i = 0; i < W / ARROW_SPACER; i++) {
    for (int j = 0; j < H / ARROW_SPACER; j++) {
      this->draw_arrow(arrow_data[i][j]);
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_corner_velocity_arrow(
    const Fluid<H, W>& fluid) {
  ArrowData arrow_data[W / ARROW_SPACER][H / ARROW_SPACER];
  for (int i = 1; i < W - 1; i += ARROW_SPACER + 1) {
    for (int j = 1; j < H - 1; j += ARROW_SPACER + 1) {
      const Cell& cell = fluid.get_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }
      int x = i * S;
      int y = (H - j - 1) * S;
      auto velocity = cell.get_velocity();
      auto vel_x = velocity.get_x();
      auto vel_y = velocity.get_y();
      auto angle = std::atan2(vel_y, vel_x);
      auto length = std::sqrt(vel_x * vel_x + vel_y * vel_y);
      arrow_data[i / ARROW_SPACER][j / ARROW_SPACER] =
          this->make_arrow_data(x, y, length, angle);
    }
  }
  for (int i = 0; i < W / ARROW_SPACER; i++) {
    for (int j = 0; j < H / ARROW_SPACER; j++) {
      this->draw_arrow(arrow_data[i][j]);
    }
  }
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_velocity_arrows(
    const Fluid<H, W>& fluid) {
#if DRAW_CORNER_ARROW
  SDL_SetRenderDrawColor(renderer, CORNER_ARROW_COLOR);
  this->update_corner_velocity_arrow(fluid);
#endif
#if DRAW_CENTER_ARROW
  SDL_SetRenderDrawColor(renderer, CENTER_ARROW_COLOR);
  this->update_center_velocity_arrow(fluid);
#endif
#if DRAW_HORIZONTAL_EDGE_ARROW
  SDL_SetRenderDrawColor(renderer, HORIZONTAL_EDGE_ARROW_COLOR);
  this->update_horizontal_edge_velocity_arrow(fluid);
#endif
#if DRAW_VERTICAL_EDGE_ARROW
  SDL_SetRenderDrawColor(renderer, VERTICAL_EDGE_ARROW_COLOR);
  this->update_vertical_edge_velocity_arrow(fluid);
#endif
}

template <int H, int W, int S>
inline void GraphicsHandler<H, W, S>::update_traces(const Fluid<H, W>& fluid,
                                                    float d_t) {
  std::array<SDL_Point, TRACE_LENGTH> traces[W / TRACE_SPACER]
                                            [H / TRACE_SPACER];
  for (int i = 1; i < W - 1; i += TRACE_SPACER) {
    for (int j = 1; j < H - 1; j += TRACE_SPACER) {
      std::array<SDL_Point, TRACE_LENGTH> points = fluid.trace(i, j, d_t);
      traces[i / TRACE_SPACER][j / TRACE_SPACER] = points;
    }
  }

  for (int i = 0; i < W / TRACE_SPACER; i++) {
    for (int j = 0; j < H / TRACE_SPACER; j++) {
      SDL_RenderDrawLines(renderer, traces[i][j].data(), TRACE_LENGTH);
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update(const Fluid<H, W>& fluid, float d_t) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels.data(),
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