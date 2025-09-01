#include "SDL_mouse.h"
#include "config.hpp"
#include "config_parser.hpp"
#include "mouse.cuh"

Mouse::Mouse()
    : is_down(false),
      wheel_value(INTERACTIVE_STARTING_SPEED),
      wheel_changed(false) {}

void Mouse::update(SDL_Event event) {
  this->wheel_changed = false;
  switch (event.type) {
    case SDL_MOUSEBUTTONDOWN: {
      this->is_down = true;
      this->button = event.button.button;
      break;
    }
    case SDL_MOUSEBUTTONUP: {
      this->is_down = false;
      break;
    }
    case SDL_MOUSEWHEEL: {
      int new_wheel_value = this->wheel_value + event.wheel.y * 10;
      if (new_wheel_value >= 0) {
        this->wheel_changed = true;
        this->wheel_value = new_wheel_value;
      }
      break;
    }
    case SDL_MOUSEMOTION: {
      int x = event.button.x;
      int y = event.button.y;
      this->position.set_x(x);
      this->position.set_y(y);
      break;
    }
  }
}

UserAction Mouse::make_user_action(Config& config) {
  UserAction action = {
      .click_action = MouseClickAction::NOTHING,
      .scroll_action = MouseScrollAction::NOTHING,
      .wheel_value = this->wheel_value,
  };
  action.position.set_x(this->position.get_x());
  action.position.set_y(config.sim.height - 1 -
                        (this->position.get_y() / config.sim.cell_pixel_size));
  if (!this->is_down) {
    return action;
  }
  switch (this->button) {
    case SDL_BUTTON_RIGHT:
      action.click_action = config.sim.mouse.right_click_action;
      break;
    case SDL_BUTTON_MIDDLE:
      action.click_action = config.sim.mouse.middle_click_action;
      break;
    case SDL_BUTTON_LEFT:
      action.click_action = config.sim.mouse.left_click_action;
      break;
  }
  if (this->wheel_changed) {
    action.scroll_action = config.sim.mouse.scroll_action;
  };
  return action;
}