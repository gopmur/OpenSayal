#include "mouse.hpp"
#include <iostream>

Mouse::Mouse() : is_down(false) {}

void Mouse::update(SDL_Event event) {
  switch (event.type) {
  case SDL_MOUSEBUTTONDOWN:
    this->is_down = true;
    this->button = event.button.button;
    break;
  case SDL_MOUSEBUTTONUP:
    this->is_down = false;
    break;

  case SDL_MOUSEMOTION:
    int x = event.button.x;
    int y = event.button.y;
    this->position.set_x(x);
    this->position.set_y(y);
  }
}

Source Mouse::make_source(int fluid_height) {
  auto position = this->position;
  position.set_y(fluid_height - position.get_y());
  double smoke = this->button == 1 ? 1 : 0;
  double velocity = this->button == 2 ? -SOURCE_SPEED : SOURCE_SPEED; 
  Source source = {
      .active = this->is_down,
      .smoke = smoke,
      .velocity = velocity,
      .position = position,
  };
  return source;
}