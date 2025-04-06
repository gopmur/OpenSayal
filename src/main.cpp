#include <cmath>
#include <cstdint>
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
#define CELL_SIZE 10

float pixelData[HEIGHT][WIDTH];

// Generate some example data (gradient effect)
void generatePixelData() {
  static int offset = 0;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      if (x + offset < WIDTH) {
        pixelData[y][x + offset] =
            static_cast<float>(WIDTH - x - 1) / WIDTH;  // Gradient from 0 to 1
      } else {
        pixelData[y][x + offset - WIDTH] =
            static_cast<float>(WIDTH - x - 1) / WIDTH;
      }
    }
  }

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

  

  return {window, renderer, texture};
}

void draw_arrow(SDL_Renderer* renderer,
                uint32_t x,
                uint32_t y,
                float length,
                float angle,
                float arrow_head_length = 5,
                float arrow_head_angle = M_PI / 6) {
  int32_t x_offset = length * std::cos(angle);
  int32_t y_offset = -length * std::sin(angle);
  uint32_t x2 = x + x_offset;
  uint32_t y2 = y + y_offset;
  SDL_RenderDrawLine(renderer, x, y, x2, y2);
  
  uint32_t arrow_x_offset = -arrow_head_length * std::cos(angle + arrow_head_angle);
  uint32_t arrow_y_offset = arrow_head_length * std::sin(angle + arrow_head_angle);
  uint32_t arrow_x2 = x2 + arrow_x_offset;
  uint32_t arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);
  
  arrow_x_offset = -arrow_head_length * std::cos(arrow_head_angle - angle);
  arrow_y_offset = -arrow_head_length * std::sin(arrow_head_angle - angle);
  arrow_x2 = x2 + arrow_x_offset;
  arrow_y2 = y2 + arrow_y_offset;
  SDL_RenderDrawLine(renderer, x2, y2, arrow_x2, arrow_y2);
}

int main() {
  auto [window, renderer, texture] = setup();
  Logger::static_log("Window created successfully");

  uint32_t* pixels = new uint32_t[WIDTH * HEIGHT];
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
        uint8_t color =
            static_cast<uint8_t>(pixelData[y][x] * 255.0f);  // Scale to 0-255
        pixels[y * WIDTH + x] =
            SDL_MapRGBA(format, color, color, color, 255);  // Grayscale
      }
    }

    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(uint32_t));

    // Render the texture
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    static float angle = 0;
    draw_arrow(renderer, 100, 100, 30, angle);
    SDL_Delay(5);
    angle += 0.01;

    SDL_RenderPresent(renderer);
  }

  delete[] pixels;
  SDL_FreeFormat(format);
  cleanup(window, renderer, texture);

  return EXIT_SUCCESS;
}
