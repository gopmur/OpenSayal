#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>

#include "SDL_events.h"

#include "config.hpp"
#include "fluid.hpp"
#include "graphics_handler.hpp"
#include "logger.hpp"
#include "platform_setup.hpp"
#include "mouse.hpp"

void setup(int argc, char* argv[]) {
  if (argc > 2 or
      (argc == 2 and (std::strcmp(argv[1], "--save-report") != 0 and
                      std::strcmp(argv[1], "--compare-performance") != 0))) {
    Logger::error(
        "usage: fluid-simulator [--save-report] [--compare-performance]");
    exit(1);
  }

  omp_set_dynamic(0);
  omp_set_num_threads(THREAD_COUNT);
  Logger::static_debug(
      std::format("requested {} threads from OpenMP", THREAD_COUNT));

  if (argc == 2) {
    Logger::init(std::strcmp(argv[1], "--save-report") == 0,
                 std::strcmp(argv[1], "--compare-performance") == 0);
  } else {
    Logger::init(0, 0);
  }

  setup_platform();
}

int main(int argc, char* argv[]) {
  setup(argc, argv);

  GraphicsHandler<FLUID_HEIGHT, FLUID_WIDTH, CELL_SIZE>* graphics =
      new GraphicsHandler<FLUID_HEIGHT, FLUID_WIDTH, CELL_SIZE>(
          ARROW_HEAD_LENGTH, ARROW_HEAD_ANGLE, ARROW_DISABLE_THRESH_HOLD);
  Fluid<FLUID_HEIGHT, FLUID_WIDTH>* fluid =
      new Fluid<FLUID_HEIGHT, FLUID_WIDTH>(PROJECTION_O, PROJECTION_N,
                                           CELL_SIZE);
  SDL_Event event;
  Mouse mouse;

  {
    using namespace std::chrono;
    bool is_running = true;
    std::optional<time_point<high_resolution_clock>> prev_time = std::nullopt;
    clock_t prev_clock = 0;
    uint64_t work = 0;
    while (is_running) {
      while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_QUIT) {
          is_running = false;
          break;
        }
        mouse.update(event);
      }
      auto now = high_resolution_clock::now();
      if (not prev_time.has_value()) {
        prev_time = now;
      }
      auto passed_time = now - prev_time.value();
      auto passed_time_ns = duration_cast<nanoseconds>(passed_time);
      auto nano_seconds = passed_time_ns.count();

      float d_t = 0;
      if (nano_seconds != 0) {
        d_t = static_cast<float>(nano_seconds) / 1'000'000'000;
        auto fps = static_cast<uint32_t>(1 / d_t);
        Logger::log_fps(d_t, work);
      }

      auto source = mouse.make_source(FLUID_HEIGHT, CELL_SIZE);

#if USE_REAL_TIME
      d_t *= REAL_TIME_MULTIPLIER;
#else
      d_t = D_T;
#endif
      prev_clock = std::clock();
      fluid->update(source, d_t);
      graphics->update(*fluid, d_t);
      work = std::clock() - prev_clock;
      prev_time = now;
    }
  }

  Logger::report();

  return EXIT_SUCCESS;
}
