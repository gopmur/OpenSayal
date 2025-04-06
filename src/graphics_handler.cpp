#include "graphics_handler.hpp"
#include <iostream>
#include "SDL.h"

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