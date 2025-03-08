#include <cstdlib>
#include <iostream>

#include "logger.hpp"

bool Logger::is_dyn_debug = false;

void Logger::init() {
  auto debug_env_var = getenv("FLUID_SIM_DEBUG");
  if (debug_env_var != NULL) {
    Logger::is_dyn_debug = true;
  }
}

void Logger::dyn_log(std::string message) {
  if (is_dyn_debug) {
    std::cout << "Log: " << message << std::endl;
  }
}

void Logger::static_log(std::string message) {
#ifndef NDEBUG
  std::cout << "Log: " << message << std::endl;
#endif
}