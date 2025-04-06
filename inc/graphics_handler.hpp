#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "SDL.h"
#include "SDL_render.h"

#include "fluid.hpp"
#include "logger.hpp"

template <uint32_t H, uint32_t W, uint32_t S>
class GraphicsHandler {
 private:
  float arrow_head_length;
  float arrow_head_angle;
  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* fluid_texture;
  SDL_PixelFormat* format;
  std::array<std::array<uint32_t, W>, H> fluid_pixels;

  void draw_arrow(uint32_t x,
                  uint32_t y,
                  float length,
                  float angle,
                  float arrow_head_length,
                  float arrow_head_angle);
  void update_fluid_pixels(const Fluid<H, W>& fluid);
  void cleanup();

 public:
  GraphicsHandler(float arrow_head_length, float arrow_head_angle);
  ~GraphicsHandler();
  void update(const Fluid<H, W>& fluid);
};

template <uint32_t H, uint32_t W, uint32_t S>
GraphicsHandler<H, W, S>::GraphicsHandler(float arrow_head_length,
                                          float arrow_head_angle)
    : arrow_head_angle(arrow_head_angle), arrow_head_length(arrow_head_length) {
  this->window = nullptr;
  this->renderer = nullptr;
  this->fluid_texture = nullptr;
  this->format = nullptr;

  uint32_t window_height = S * H;
  uint32_t window_width = S * W;

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

  Logger::static_log("Graphics initialized successfully");
}

template <uint32_t H, uint32_t W, uint32_t S>
GraphicsHandler<H, W, S>::~GraphicsHandler() {
  this->cleanup();
}

template <uint32_t H, uint32_t W, uint32_t S>
void GraphicsHandler<H, W, S>::cleanup() {
  Logger::static_log("Cleaning up graphics");
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

template <uint32_t H, uint32_t W, uint32_t S>
void GraphicsHandler<H, W, S>::draw_arrow(uint32_t x,
                                          uint32_t y,
                                          float length,
                                          float angle,
                                          float arrow_head_length,
                                          float arrow_head_angle) {
  int32_t x_offset = length * std::cos(angle);
  int32_t y_offset = -length * std::sin(angle);
  uint32_t x2 = x + x_offset;
  uint32_t y2 = y + y_offset;
  SDL_RenderDrawLine(renderer, x, y, x2, y2);

  int32_t arrow_x_offset =
      -arrow_head_length * std::cos(angle + arrow_head_angle);
  int32_t arrow_y_offset =
      arrow_head_length * std::sin(angle + arrow_head_angle);
  uint32_t arrow_x2 = x2 + arrow_x_offset;
  uint32_t arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);

  arrow_x_offset = -arrow_head_length * std::cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * std::sin(arrow_head_angle - angle);
  arrow_x2 = x2 + arrow_x_offset;
  arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);
}

template <uint32_t H, uint32_t W, uint32_t S>
void GraphicsHandler<H, W, S>::update_fluid_pixels(const Fluid<H, W>& fluid) {
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      auto cell = fluid.get_cell(i, j);
      auto smoke_density = cell.get_smoke().get_density();
      uint8_t color =
          static_cast<uint8_t>(smoke_density * 255.0f);  // Scale to 0-255
      this->fluid_pixels[i][j] =
          SDL_MapRGBA(this->format, color, color, color, 255);  // Grayscale
    }
  }
}

template <uint32_t H, uint32_t W, uint32_t S>
void GraphicsHandler<H, W, S>::update(const Fluid<H, W>& fluid) {
  this->update_fluid_pixels(fluid);
  SDL_UpdateTexture(this->fluid_texture, NULL, this->fluid_pixels.data(),
                    W * sizeof(uint32_t));

  // Render the texture
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, this->fluid_texture, NULL, NULL);
  static float angle = 0;
  this->draw_arrow(100, 100, 50, angle, this->arrow_head_length,
                   this->arrow_head_angle);
  SDL_Delay(5);
  angle += 0.01;

  SDL_RenderPresent(renderer);
}