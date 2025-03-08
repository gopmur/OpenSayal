#include <cstdlib>
#include <iostream>

#include "SDL.h"

#include "logger.hpp"
#include "platform_setup.hpp"

SDL_Window* setup() {
  Logger::init();
  setup_platform();

  auto sdl_status = SDL_Init(SDL_INIT_VIDEO);
  bool sdl_has_error = sdl_status < 0;
  if (sdl_has_error) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Video initialization failed: " << sdl_error_message
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto window =
      SDL_CreateWindow("Fluid simulation", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_SHOWN);
  sdl_has_error = window == NULL;
  if (sdl_has_error) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Window creation failed: " << sdl_error_message << std::endl;
    exit(EXIT_FAILURE);
  }
  return window;
}

void cleanup(SDL_Window* window) {
  SDL_DestroyWindow(window);
  SDL_Quit();
}

int main() {
  auto window = setup();
  Logger::static_log("Window created successfully");

  SDL_Surface* surface = SDL_GetWindowSurface(window);
  if (surface) {
    SDL_UpdateWindowSurface(window);
  }

  SDL_Delay(5000);

  cleanup(window);

  return EXIT_SUCCESS;
}