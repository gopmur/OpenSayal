#pragma once

#include <array>
#include <iostream>
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

  inline static int avg_fps();
  inline static int avg_d_t();
  inline static void add_fps(int fps);
  inline static void add_d_t(int d_t);

 public:
  inline static void init();
  inline static void dyn_debug(std::string message);
  inline static void static_debug(std::string message);
  inline static void log_fps(float d_t);
};

inline void Logger::init() {
  auto debug_env_var = getenv("FLUID_SIM_DEBUG");
  if (debug_env_var != NULL) {
    Logger::is_dyn_debug = true;
  }
}

inline void Logger::dyn_debug(std::string message) {
  if (Logger::is_dyn_debug) {
    std::cout << "Log: " << message << std::endl;
  }
}

inline void Logger::static_debug(std::string message) {
#ifndef NDEBUG
  std::cout << "Log: " << message << std::endl;
#endif
}

inline void Logger::add_fps(int fps) {
  Logger::fps_memo[Logger::fps_memo_index] = fps;
  if (Logger::fps_memo_length < FPS_AVG_SIZE) {
    Logger::fps_memo_length++;
  }
  Logger::fps_memo_index++;
  Logger::fps_memo_index %= FPS_AVG_SIZE;
}

inline void Logger::add_d_t(int d_t) {
  Logger::d_t_memo[Logger::d_t_memo_index] = d_t;
  if (Logger::d_t_memo_length < FPS_AVG_SIZE) {
    Logger::d_t_memo_length++;
  }
  Logger::d_t_memo_index++;
  Logger::d_t_memo_index %= FPS_AVG_SIZE;
}

inline int Logger::avg_fps() {
  float fps_avg = 0;
  for (int i = 0; i < Logger::fps_memo_length; i++) {
    fps_avg += Logger::fps_memo[i];
  }
  fps_avg /= Logger::fps_memo_length;
  return static_cast<int>(fps_avg);
}

inline int Logger::avg_d_t() {
  int d_t_avg = 0;
  for (int i = 0; i < Logger::d_t_memo_length; i++) {
    d_t_avg += Logger::d_t_memo[i];
  }
  d_t_avg /= Logger::d_t_memo_length;
  return static_cast<int>(d_t_avg);
}

inline void Logger::log_fps(float d_t) {
  // if (fps_memo_length == FPS_AVG_SIZE) return;
  auto fps = static_cast<int>(1 / d_t);
  Logger::add_fps(fps);
  Logger::add_d_t(d_t * 1'000'000);
  // if (fps_memo_length == FPS_AVG_SIZE) {
  auto fps_avg = Logger::avg_fps();
  auto d_t_avg = Logger::avg_d_t();
  if (d_t_memo_length == FPS_AVG_SIZE) {
    std::cout << "\033[1;32m";
  }
  std::cout << "FPS: " << fps_avg << ", DT: " << d_t_avg << "um" << std::endl;
  if (d_t_memo_length == FPS_AVG_SIZE) {
    std::cout << "\033[0m";
  }
  // }
}