#include <cstdlib>

#include "config.hpp"
#include "logger.hpp"

bool Logger::is_dyn_debug = false;
int Logger::memo_length = 0;
int Logger::memo_index = 0;
bool Logger::first_add = true;
int Logger::d_t_memo[FPS_AVG_SIZE];
uint64_t Logger::work_memo[FPS_AVG_SIZE];
bool Logger::file_wrote = false;
int Logger::stable_avg_d_t = 0;
int Logger::fps_log_number = 0;
bool Logger::save_report = false;
bool Logger::stable_d_t_stored = false;
bool Logger::compare_performance = false;