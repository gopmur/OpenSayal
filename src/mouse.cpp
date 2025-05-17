#include "mouse.hpp"
#include <iostream>

Mouse::Mouse() : is_down(false), wheel_value(SOURCE_SPEED) {}

void Mouse::update(SDL_Event event) {
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

Source Mouse::make_source(int fluid_height, int cell_size) {
  auto position = this->position / cell_size;
  position.set_y(fluid_height - position.get_y());
  double smoke = this->button == 1 ? 1 : 0;
  double velocity = this->button == 2 ? -this->wheel_value : this->wheel_value;
  Source source = {
      .active = this->is_down,
      .smoke = smoke,
      .velocity = velocity,
      .position = position,
  };
  return source;
}