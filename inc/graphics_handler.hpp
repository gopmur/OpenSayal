#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "SDL.h"
#include "SDL_render.h"

#include "config.hpp"
#include "fluid.hpp"
#include "helper.hpp"
#include "logger.hpp"

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

  void draw_arrow(int x, int y, float length, float angle);
  void update_fluid_pixels(const Fluid<H, W>& fluid);
  void update_velocity_arrows(const Fluid<H, W>& fluid);
  void update_center_velocity_arrow(const Fluid<H, W>& fluid);
  void update_horizontal_edge_velocity_arrow(const Fluid<H, W>& fluid);
  void update_vertical_edge_velocity_arrow(const Fluid<H, W>& fluid);
  void update_corner_velocity_arrow(const Fluid<H, W>& fluid);
  void cleanup();

 public:
  GraphicsHandler(float arrow_head_length,
                  float arrow_head_angle,
                  float arrow_disable_thresh_hold);
  ~GraphicsHandler();
  void update(const Fluid<H, W>& fluid);
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
    std::cerr << "Video initialization failed: " << sdl_error_message
              << std::endl;
    exit(EXIT_FAILURE);
  }

  this->window = SDL_CreateWindow("Fluid simulation", SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED, window_width,
                                  window_height, SDL_WINDOW_SHOWN);
  if (this->window == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cerr << "Window creation failed: " << sdl_error_message << std::endl;
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->renderer =
      SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
  if (this->renderer == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Renderer creation failed: " << sdl_error_message << std::endl;
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->fluid_texture =
      SDL_CreateTexture(this->renderer, SDL_PIXELFORMAT_RGBA8888,
                        SDL_TEXTUREACCESS_STREAMING, W, H);
  if (this->fluid_texture == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Texture creation failed: " << sdl_error_message << std::endl;
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  this->format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);
  if (this->format == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Format allocation failed: " << sdl_error_message << std::endl;
    this->cleanup();
    exit(EXIT_FAILURE);
  }

  Logger::static_debug("Graphics initialized successfully");
}

template <int H, int W, int S>
GraphicsHandler<H, W, S>::~GraphicsHandler() {
  this->cleanup();
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::cleanup() {
  Logger::static_debug("Cleaning up graphics");
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
void GraphicsHandler<H, W, S>::draw_arrow(int x,
                                          int y,
                                          float length,
                                          float angle) {
  length *= ARROW_LENGTH_MULTIPLIER;
  int32_t x_offset = length * std::cos(angle);
  int32_t y_offset = -length * std::sin(angle);
  int x2 = x + x_offset;
  int y2 = y + y_offset;
  SDL_RenderDrawLine(renderer, x, y, x2, y2);
  int32_t arrow_x_offset =
      -arrow_head_length * std::cos(angle + arrow_head_angle);
  int32_t arrow_y_offset =
      arrow_head_length * std::sin(angle + arrow_head_angle);
  int arrow_x2 = x2 + arrow_x_offset;
  int arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);

  arrow_x_offset = -arrow_head_length * std::cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * std::sin(arrow_head_angle - angle);
  arrow_x2 = x2 + arrow_x_offset;
  arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_fluid_pixels(const Fluid<H, W>& fluid) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      const Cell& cell = fluid.get_cell(i, j);

      int x = i;
      int y = H - j - 1;

      if (cell.is_solid()) {
        this->fluid_pixels[y][x] = SDL_MapRGBA(this->format, 80, 80, 80, 255);
      } else {
        auto smoke_density = cell.get_density();
        uint8_t color = 255 - static_cast<uint8_t>(smoke_density * 255);
        this->fluid_pixels[y][x] =
            SDL_MapRGBA(this->format, 255, color, color, 255);
      }
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_center_velocity_arrow(
    const Fluid<H, W>& fluid) {
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
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
      if (length > this->arrow_disable_thresh_hold) {
        this->draw_arrow(x, y, length, angle);
      }
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_horizontal_edge_velocity_arrow(
    const Fluid<H, W>& fluid) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
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
      if (length > this->arrow_disable_thresh_hold) {
        this->draw_arrow(x, y, length, angle);
      }
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_vertical_edge_velocity_arrow(
    const Fluid<H, W>& fluid) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
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
      if (length > this->arrow_disable_thresh_hold) {
        this->draw_arrow(x, y, length, angle);
      }
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_corner_velocity_arrow(
    const Fluid<H, W>& fluid) {
  for (int i = 1; i < W - 1; i++) {
    for (int j = 1; j < H - 1; j++) {
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
      if (length > this->arrow_disable_thresh_hold) {
        this->draw_arrow(x, y, length, angle);
      }
    }
  }
}

template <int H, int W, int S>
void GraphicsHandler<H, W, S>::update_velocity_arrows(
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
void GraphicsHandler<H, W, S>::update(const Fluid<H, W>& fluid) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels.data(),
                    W * sizeof(int));

  // Render the texture
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, this->fluid_texture, NULL, NULL);

  this->update_velocity_arrows(fluid);

  SDL_RenderPresent(renderer);
}