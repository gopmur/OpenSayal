#pragma once

#include <array>
#include <string>

#include "config.hpp"

class Logger {
 private:
  static bool is_dyn_debug;
  static std::array<int, FPS_AVG_SIZE> fps_memo;
  static int fps_memo_length;
  static int fps_memo_index;
  static int d_t_memo_length;
  static int d_t_memo_index;
  static std::array<int, FPS_AVG_SIZE> d_t_memo;

  static int avg_fps();
  static int avg_d_t();
  static void add_fps(int fps);
  static void add_d_t(int d_t);

 public:
  static void init();
  static void dyn_debug(std::string message);
  static void static_debug(std::string message);
  static void log_fps(float d_t);
};