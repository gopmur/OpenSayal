#pragma once

#include <optional>

#include "SDL_events.h"

#include "helper.hpp"

class Mouse {
private:
  bool is_down;
  Vector2d<int> position;

public:
  Mouse();
  void update(SDL_Event event);
  std::optional<Vector2d<int>> get_pressed_position();
};