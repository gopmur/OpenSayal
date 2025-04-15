#include <cstdlib>

#include "config.hpp"
#include "logger.hpp"

bool Logger::is_dyn_debug = false;
int Logger::fps_memo_length = 0;
int Logger::fps_memo_index = 0;
int Logger::d_t_memo_length = 0;
int Logger::d_t_memo_index = 0;
std::array<int, FPS_AVG_SIZE> Logger::fps_memo;
std::array<int, FPS_AVG_SIZE> Logger::d_t_memo;