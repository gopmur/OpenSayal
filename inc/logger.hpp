#pragma once

#include <string>

class Logger {
  private:
    static bool is_dyn_debug;
  public:
    static void init();
    static void dyn_log(std::string message);
    static void static_log(std::string message);
};