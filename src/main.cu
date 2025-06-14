#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <optional>

#include "SDL_events.h"
#include "json.hpp"

#include "config_parser.hpp"
#include "fluid.cuh"
#include "graphics_handler.cuh"
#include "logger.hpp"
#include "mouse.cuh"
#include "platform_setup.hpp"

void setup(int argc, char* argv[], Config config) {
  if (argc > 2 or
      (argc == 2 and (std::strcmp(argv[1], "--save-report") != 0 and
                      std::strcmp(argv[1], "--compare-performance") != 0))) {
    Logger::error(
        "usage: fluid-simulator [--save-report] [--compare-performance]");
    exit(1);
  }

  omp_set_dynamic(0);
  omp_set_num_threads(config.thread.openMP.thread_count);
  Logger::static_debug(std::format("requested {} threads from OpenMP",
                                   config.thread.openMP.thread_count));

  if (argc == 2) {
    Logger::init(std::strcmp(argv[1], "--save-report") == 0,
                 std::strcmp(argv[1], "--compare-performance") == 0);
  } else {
    Logger::init(0, 0);
  }

  setup_platform();
}

int main(int argc, char* argv[]) {
  auto config_parser = ConfigParser();
  auto config = config_parser.parse();

  setup(argc, argv, config);

  GraphicsHandler graphics(config);
  Fluid fluid(config);
  SDL_Event event;
  Mouse mouse;
  Source source;

  {
    using namespace std::chrono;
    bool is_running = true;
    std::optional<time_point<high_resolution_clock>> prev_time = std::nullopt;
    clock_t prev_clock = 0;
    uint64_t work = 0;
    while (true) {
      while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_QUIT) {
          is_running = false;
          break;
        }
        if (config.sim.enable_interactive) {
          mouse.update(event);
        }
      }
      if (!is_running) {
        break;
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
      if (config.sim.enable_interactive) {
        source =
            mouse.make_source(config.sim.height, config.sim.cell_pixel_size);
      }
      if (config.sim.time.enable_read_time) {
        d_t *= config.sim.time.real_time_multiplier;
      } else {
        d_t = config.sim.time.d_t;
      }
      prev_clock = std::clock();
      fluid.update(source, d_t);
      graphics.update(fluid, d_t);
      work = std::clock() - prev_clock;
      prev_time = now;
    }
  }

  Logger::report();

  return EXIT_SUCCESS;
}
