#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <tuple>

#include "SDL.h"
#include "SDL_error.h"
#include "SDL_events.h"
#include "SDL_render.h"
#include "SDL_timer.h"
#include "SDL_video.h"

#include "logger.hpp"
#include "platform_setup.hpp"

#define HEIGHT 100
#define WIDTH 100

float pixelData[HEIGHT][WIDTH];

// Generate some example data (gradient effect)
void generatePixelData() {
  static int offset = 0;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      if (x + offset < WIDTH) {
        pixelData[y][x + offset] = static_cast<float>(x) / WIDTH;  // Gradient from 0 to 1
      }
      else {
        pixelData[y][x + offset - WIDTH] = static_cast<float>(x) / WIDTH;
      }
    }
  }
  offset++;
  offset %= WIDTH;
}

void cleanup(SDL_Window* window, SDL_Renderer* renderer, SDL_Texture* texture) {
  if (window != nullptr) {
    SDL_DestroyWindow(window);
  }
  if (renderer != nullptr) {
    SDL_DestroyRenderer(renderer);
  }
  if (texture != nullptr) {
    SDL_DestroyTexture(texture);
  }
  SDL_Quit();
}

std::tuple<SDL_Window*, SDL_Renderer*, SDL_Texture*> setup() {
  Logger::init();
  setup_platform();

  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* texture = nullptr;

  auto sdl_status = SDL_Init(SDL_INIT_VIDEO);
  if (sdl_status < 0) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Video initialization failed: " << sdl_error_message
              << std::endl;
    exit(EXIT_FAILURE);
  }

  window =
      SDL_CreateWindow("Fluid simulation", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, WIDTH * 10, HEIGHT * 10, SDL_WINDOW_SHOWN);
  if (window == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Window creation failed: " << sdl_error_message << std::endl;
    cleanup(window, renderer, texture);
    exit(EXIT_FAILURE);
  }

  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (renderer == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Renderer creation failed: " << sdl_error_message << std::endl;
    cleanup(window, renderer, texture);
    exit(EXIT_FAILURE);
  }

  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                              SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
  if (texture == nullptr) {
    auto sdl_error_message = SDL_GetError();
    std::cout << "Texture creation failed: " << sdl_error_message << std::endl;
    cleanup(window, renderer, texture);
    exit(EXIT_FAILURE);
  }

  return {window, renderer, texture};
}

// void main_loop(SDL_Window* window,
//                SDL_Renderer* renderer,
//                SDL_Texture* texture) {
//   SDL_UpdateWindowSurface(window);
// }

int main() {
  auto [window, renderer, texture] = setup();
  Logger::static_log("Window created successfully");

  Uint32* pixels = new Uint32[WIDTH * HEIGHT];
  SDL_PixelFormat* format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);

  SDL_Event event;
  bool is_running = true;
  while (is_running) {
    while (SDL_PollEvent(&event) != 0) {
      if (event.type == SDL_QUIT) {
        is_running = false;
      }
    }
    generatePixelData();
    for (int y = 0; y < HEIGHT; ++y) {
      for (int x = 0; x < WIDTH; ++x) {
        Uint8 color =
            static_cast<Uint8>(pixelData[y][x] * 255.0f);  // Scale to 0-255
        pixels[y * WIDTH + x] =
            SDL_MapRGBA(format, color, color, color, 255);  // Grayscale
      }
    }

    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(Uint32));

    SDL_RenderSetScale(renderer, 10, 10);

    // Render the texture
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
  }

  delete[] pixels;
  SDL_FreeFormat(format);
  cleanup(window, renderer, texture);

  return EXIT_SUCCESS;
}