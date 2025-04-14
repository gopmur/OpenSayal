#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "config.hpp"
#include "logger.hpp"

bool Logger::is_dyn_debug = false;
uint32_t Logger::fps_memo_length = 0;
uint32_t Logger::fps_memo_index = 0;
std::array<uint32_t, FPS_AVG_SIZE> Logger::fps_memo;

void Logger::init() {
  auto debug_env_var = getenv("FLUID_SIM_DEBUG");
  if (debug_env_var != NULL) {
    Logger::is_dyn_debug = true;
  }
}

void Logger::dyn_debug(std::string message) {
  if (Logger::is_dyn_debug) {
    std::cout << "Log: " << message << std::endl;
  }
}

void Logger::static_debug(std::string message) {
#ifndef NDEBUG
  std::cout << "Log: " << message << std::endl;
#endif
}

void Logger::add_fps(uint32_t fps) {
  Logger::fps_memo[Logger::fps_memo_index] = fps;
  if (Logger::fps_memo_length < FPS_AVG_SIZE) {
    Logger::fps_memo_length++;
  }
  Logger::fps_memo_index++;
  Logger::fps_memo_index %= FPS_AVG_SIZE;
}

uint32_t Logger::avg_fps() {
  float fps_avg = 0;
  for (uint32_t i = 0; i < Logger::fps_memo_length; i++) {
    fps_avg += Logger::fps_memo[i];
  }
  fps_avg /= Logger::fps_memo_length;
  return static_cast<uint32_t>(fps_avg);
}

void Logger::log_fps(float d_t) {
  if (fps_memo_length == FPS_AVG_SIZE) return;
  auto fps = static_cast<uint32_t>(1 / d_t);
  Logger::add_fps(fps);
  if (fps_memo_length == FPS_AVG_SIZE) {
    auto fps_avg = Logger::avg_fps();
    std::cout << "FPS: " << fps_avg << std::endl;
  }
}