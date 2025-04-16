#include <cstdlib>
#include <format>

#include "logger.hpp"

void setup_platform() {
#ifdef __linux__

  Logger::static_debug("system is linux");

  auto session_type = getenv("XDG_SESSION_TYPE");

  Logger::static_debug(std::format("XDG_SESSION_TYPE: {}", session_type));
  setenv("SDL_VIDEODRIVER", session_type, true);
  auto sld_video_driver = getenv("SDL_VIDEODRIVER");
  Logger::static_debug(std::format("SDL_VIDEODRIVER: {}", sld_video_driver));
#endif
}