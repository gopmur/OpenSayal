#include "mouse.hpp"

Mouse::Mouse() : is_down(false) {}

void Mouse::update(SDL_Event event) {
  switch (event.type) {
  case SDL_MOUSEBUTTONDOWN:
    this->is_down = true;
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

std::optional<Vector2d<int>> Mouse::get_pressed_position() {
  if (!this->is_down) {
    return std::nullopt;
  }
  return this->position;
}