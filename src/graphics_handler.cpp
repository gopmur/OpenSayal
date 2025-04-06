#include <cmath>
#include <iostream>

#include "SDL.h"

#include "graphics_handler.hpp"

GraphicsHandler::GraphicsHandler(uint32_t fluid_cell_size,
                                 uint32_t fluid_height,
                                 uint32_t fluid_width) {
  this->window = nullptr;
  this->renderer = nullptr;
  this->fluid_texture = nullptr;

  uint32_t window_height = fluid_cell_size * fluid_height;
  uint32_t window_width = fluid_cell_size * fluid_width;

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
                        SDL_TEXTUREACCESS_STREAMING, fluid_width, fluid_height);
  if (this->fluid_texture == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Texture creation failed: " << sdl_error_message << std::endl;
    this->cleanup();
    exit(EXIT_FAILURE);
  }
}

GraphicsHandler::~GraphicsHandler() {
  this->cleanup();
}

void GraphicsHandler::draw_arrow(uint32_t x,
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