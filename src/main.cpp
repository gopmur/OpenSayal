#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "SDL.h"
#include "SDL_events.h"
#include "SDL_render.h"
#include "SDL_video.h"

#include "config.hpp"
#include "fluid.hpp"
#include "graphics_handler.hpp"
#include "logger.hpp"
#include "platform_setup.hpp"

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

void setup() {
  Logger::init();
  setup_platform();
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

int main() {
  setup();

  GraphicsHandler<FLUID_HEIGHT, FLUID_WIDTH, CELL_SIZE> graphics(
      ARROW_HEAD_LENGTH, ARROW_HEAD_ANGLE);
  Fluid<FLUID_HEIGHT, FLUID_WIDTH> fluid(1.8, 100);

  SDL_Event event;
  bool is_running = true;
  while (is_running) {
    while (SDL_PollEvent(&event) != 0) {
      if (event.type == SDL_QUIT) {
        is_running = false;
      }
    }
    graphics.update(fluid);
  }

  return EXIT_SUCCESS;
}
