#pragma once

#include <optional>

#include "SDL_events.h"

#include "fluid.hpp"
#include "helper.hpp"

class Mouse {
private:
  bool is_down;
  Vector2d<int> position;
  int button;
  int wheel_value;

public:
  Mouse();
  void update(SDL_Event event);
  Source make_source(int fluid_height);
};