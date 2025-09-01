#pragma once

#include "SDL_events.h"
#include "config_parser.hpp"
#include "helper.cuh"

struct UserAction {
  MouseClickAction click_action;
  MouseScrollAction scroll_action;
  Vector2d<int> position;
  int wheel_value;
};

class Mouse {
 private:
  bool is_down;
  Vector2d<int> position;
  int button;
  int wheel_value;
  bool wheel_changed;

 public:
  Mouse();
  void update(SDL_Event event);
  UserAction make_user_action(Config& config);
};
