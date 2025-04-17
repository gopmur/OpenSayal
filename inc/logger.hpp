#pragma once

#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include "config.hpp"

class Logger {
 private:
  static bool is_dyn_debug;
  static int fps_memo[FPS_AVG_SIZE];
  static uint64_t work_memo[FPS_AVG_SIZE];
  static int memo_length;
  static int memo_index;
  static int d_t_memo[FPS_AVG_SIZE];
  static bool first_add;
  static bool file_wrote;
  static int stable_avg_d_t;
  static int fps_log_number;
  static bool save_report;
  static bool stable_d_t_stored;
  static bool compare_performance;

  inline static int avg_fps();
  inline static int avg_d_t();
  inline static uint64_t avg_work();
  inline static void add_values(int fps, int d_t, uint64_t work);
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
  inline static void log_fps(float d_t, uint64_t work);
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

inline void Logger::add_values(int fps, int d_t, uint64_t work) {
  if (Logger::first_add) {
    return;
  }
  Logger::fps_memo[Logger::memo_index] = fps;
  Logger::d_t_memo[Logger::memo_index] = d_t;
  Logger::work_memo[Logger::memo_index] = work;
  if (Logger::memo_length < FPS_AVG_SIZE) {
    Logger::memo_length++;
  }
  Logger::memo_index++;
  Logger::memo_index %= FPS_AVG_SIZE;
}

inline int Logger::avg_fps() {
  float fps_avg = 0;
  for (int i = 0; i < Logger::memo_length; i++) {
    fps_avg += Logger::fps_memo[i];
  }
  if (Logger::memo_length != 0) {
    fps_avg /= Logger::memo_length;
  }
  return static_cast<int>(fps_avg);
}

inline int Logger::avg_d_t() {
  float d_t_avg = 0;
  for (int i = 0; i < Logger::memo_length; i++) {
    d_t_avg += Logger::d_t_memo[i];
  }
  if (Logger::memo_length != 0) {
    d_t_avg /= Logger::memo_length;
  }
  return static_cast<int>(d_t_avg);
}

inline uint64_t Logger::avg_work() {
  uint64_t work_avg = 0;
  for (int i = 0; i < Logger::memo_length; i++) {
    work_avg += Logger::work_memo[i];
  }
  if (Logger::memo_length != 0) {
    work_avg /= Logger::memo_length;
  }
  return static_cast<uint64_t>(work_avg);
}

inline void Logger::log_fps(float d_t, uint64_t work) {
  auto fps = static_cast<int>(1 / d_t);
  Logger::add_values(fps, d_t * 1'000'000, work);
  auto fps_avg = Logger::avg_fps();
  auto d_t_avg = Logger::avg_d_t();
  auto work_avg = Logger::avg_work();
  if (Logger::save_report and memo_length == FPS_AVG_SIZE and
      not Logger::file_wrote) {
    Logger::file_wrote = true;
    store_d_t(d_t_avg);
  };
  if (not Logger::stable_d_t_stored and memo_length == FPS_AVG_SIZE) {
    stable_d_t_stored = true;
    stable_avg_d_t = d_t_avg;
  }
  fps_log_number++;
  if (fps_log_number != FPS_LOG_SPACER)
    return;
  fps_log_number %= FPS_LOG_SPACER;
  if (memo_length == FPS_AVG_SIZE and not Logger::first_add) {
    Logger::green(std::format("FPS: {}, DT: {}us, Work: {}\n", fps_avg, d_t_avg, work_avg));
  } else if (not Logger::first_add) {
    std::cout << "FPS: " << fps_avg << ", DT: " << d_t_avg
              << "us, Work: " << work_avg << std::endl;
  }
  Logger::first_add = false;
}