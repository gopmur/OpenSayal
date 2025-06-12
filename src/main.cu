#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>

#include "SDL_events.h"

#include "config.hpp"
#include "fluid.cu"
#include "graphics_handler.cu"
#include "logger.hpp"
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
  Config config;

  config = {
      .thread =
          {
              .openMP = {.thread_count = THREAD_COUNT},
              .cuda = {.block_size_x = BLOCK_SIZE_X,
                       .block_size_y = BLOCK_SIZE_Y},
          },
      .sim =
          {
              .projection =
                  {
                      .n = PROJECTION_N,
                      .o = PROJECTION_O,
                  },
              .wind_tunnel =
                  {
                      .pipe_height = PIPE_HEIGHT,
                      .pipe_length = PIPE_LENGTH,
                      .smoke_length = SMOKE_LENGTH,
                      .speed = WIND_SPEED,
                      .smoke = WIND_SMOKE,
                  },
              .physics =
                  {
                      .g = PHYSICS_G,
                  },
              .time =
                  {
                      .d_t = D_T,
                      .enable_read_time = USE_REAL_TIME,
                      .real_time_multiplier = REAL_TIME_MULTIPLIER,
                  },
              .smoke =
                  {
                      .enable_decay = ENABLE_SMOKE_DECAY,
                      .decay_rate = SMOKE_DECAY_RATE,
                  },
              .obstacle =
                  {
                      .enable = ENABLE_CIRCLE,
                      .center_x = CIRCLE_POSITION_X,
                      .center_y = CIRCLE_POSITION_Y,
                      .radius = CIRCLE_RADIUS,
                  },
              .height = FLUID_HEIGHT,
              .width = FLUID_WIDTH,
              .cell_pixel_size = CELL_SIZE,
              .cell_size = CELL_SIZE,
              .enable_drain = !ENABLE_RIGHT_WALL,
              .enable_pressure = ENABLE_PRESSURE,
              .enable_smoke = ENABLE_SMOKE,
          },
      .fluid =
          {
              .density = FLUID_DENSITY,
              .drag_coeff = DRAG_COEFF,
          },
      .visual =
          {
              .arrows =
                  {
                      .color =
                          {
                              .r = 255,
                              .g = 255,
                              .b = 255,
                              .a = 255,
                          },
                      .enable = DRAW_CENTER_ARROW,
                      .distance = ARROW_SPACER,
                      .length_multiplier = ARROW_LENGTH_MULTIPLIER,
                      .disable_threshold = ARROW_DISABLE_THRESH_HOLD,
                      .head_length = ARROW_HEAD_LENGTH,
                  },
              .path_line =
                  {
                      .enable = ENABLE_TRACES,
                      .length = TRACE_LENGTH,
                      .color =
                          {
                              .r = 255,
                              .g = 255,
                              .b = 255,
                              .a = 255,
                          },
                      .distance = TRACE_SPACER,
                  },
          },
  };

  setup(argc, argv, config);

  GraphicsHandler graphics(config);
  Fluid fluid(config);
  SDL_Event event;

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
        }
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

      if (config.sim.time.enable_read_time) {
        d_t *= config.sim.time.real_time_multiplier;
      } else {
        d_t = config.sim.time.d_t;
      }
      prev_clock = std::clock();
      fluid.update(d_t);
      graphics.update(fluid, d_t);
      work = std::clock() - prev_clock;
      prev_time = now;
    }
  }

  Logger::report();

  return EXIT_SUCCESS;
}
