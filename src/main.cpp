#include <cmath>
#include <cstdlib>

#include "SDL_events.h"

#include "config.hpp"
#include "fluid.hpp"
#include "graphics_handler.hpp"
#include "logger.hpp"
#include "platform_setup.hpp"

void setup() {
  Logger::init();
  setup_platform();
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
