#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "config.hpp"

class Logger {
 private:
  static bool is_dyn_debug;
  static std::array<uint32_t, FPS_AVG_SIZE> fps_memo;
  static uint32_t fps_memo_length;
  static uint32_t fps_memo_index;

  static uint32_t avg_fps();
  static void add_fps(uint32_t fps);

 public:
  static void init();
  static void dyn_debug(std::string message);
  static void static_debug(std::string message);
  static void log_fps(float d_t);
};