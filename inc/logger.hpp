#pragma once

#include <array>
#include <format>
#include <fstream>
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
  static bool file_wrote;
  static int stable_avg_d_t;
  static int fps_log_number;
  static bool save_report;
  static bool stable_d_t_stored;
  static bool compare_performance;

  inline static int avg_fps();
  inline static int avg_d_t();
  inline static void add_fps(int fps);
  inline static void add_d_t(int d_t);
  inline static void store_d_t(int d_t);
  inline static int read_d_t();
  inline static void green(std::string message);
  inline static void red(std::string message);

 public:
  inline static void init(bool save_report, bool compare_performance);
  inline static void dyn_debug(std::string message);
  inline static void static_debug(std::string message);
  inline static void error(std::string message);
  inline static void warning(std::string message);
  inline static void log_fps(float d_t);
  inline static void report();
};

inline void Logger::green(std::string message) {
  std::cout << "\033[32m" << message << "\033[0m";
}

inline void Logger::red(std::string message) {
  std::cout << "\033[31m" << message << "\033[0m";
}

inline void Logger::error(std::string message) {
  std::cerr << "\033[31m" << "Error: " << message << "\033[0m" << std::endl;
}

inline void Logger::warning(std::string message) {
  std::cerr << "\033[33m" << "Warning: " << message << "\033[0m" << std::endl;
}

inline int Logger::read_d_t() {
  std::ifstream report_file(REPORT_FILE_NAME);
  int previous_d_t;
  report_file >> previous_d_t;
  return previous_d_t;
}

inline void Logger::report() {
  if (not Logger::compare_performance) {
    return;
  }
  auto current_d_t = Logger::stable_avg_d_t;
  if (not Logger::stable_d_t_stored) {
    Logger::warning(
        "d_t didn't stabilized this may result in inaccurate performance "
        "report");
    current_d_t = Logger::avg_d_t();
  }
  auto previous_d_t = read_d_t();
  float performance_gain =
      (previous_d_t - current_d_t) / static_cast<float>(previous_d_t);
  std::cout << "performance gain: ";
  if (performance_gain > 0) {
    Logger::green(std::format("{}%\n", performance_gain * 100));
  } else {
    Logger::red(std::format("{}%\n", performance_gain * 100));
  }
}

inline void Logger::store_d_t(int d_t) {
  std::ofstream report_file(REPORT_FILE_NAME);
  if (not report_file) {
    Logger::error("couldn't open report file");
  }
  report_file << d_t << std::endl;
}

inline void Logger::init(bool save_report, bool compare_performance) {
  Logger::save_report = save_report;
  Logger::compare_performance = compare_performance;
  auto debug_env_var = getenv("FLUID_SIM_DEBUG");
  if (debug_env_var != NULL) {
    Logger::is_dyn_debug = true;
  }
}

inline void Logger::dyn_debug(std::string message) {
  if (Logger::is_dyn_debug) {
    std::cout << "Debug: " << message << std::endl;
  }
}

inline void Logger::static_debug(std::string message) {
#if not defined(NDEBUG) or ALLOW_LOG_ON_RELEASE
  std::cout << "Debug: " << message << std::endl;
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
  auto fps = static_cast<int>(1 / d_t);
  Logger::add_fps(fps);
  Logger::add_d_t(d_t * 1'000'000);
  auto fps_avg = Logger::avg_fps();
  auto d_t_avg = Logger::avg_d_t();
  if (Logger::save_report and fps_memo_length == FPS_AVG_SIZE and
      not Logger::file_wrote) {
    Logger::file_wrote = true;
    store_d_t(d_t_avg);
  };
  if (not Logger::stable_d_t_stored and fps_memo_length == FPS_AVG_SIZE) {
    stable_d_t_stored = true;
    stable_avg_d_t = d_t_avg;
  }
  fps_log_number++;
  if (fps_log_number != FPS_LOG_SPACER)
    return;
  fps_log_number %= FPS_LOG_SPACER;
  if (d_t_memo_length == FPS_AVG_SIZE) {
    Logger::green(std::format("FPS: {}, DT: {}um\n", fps_avg, d_t_avg));
  } else {
    std::cout << "FPS: " << fps_avg << ", DT: " << d_t_avg << "um" << std::endl;
  }
}