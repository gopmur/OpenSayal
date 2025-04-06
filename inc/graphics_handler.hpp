#pragma once

#include "SDL_render.h"

class GraphicsHandler {
 private:
  const float ARROW_HEAD_LENGTH = 5;
  const float ARROW_HEAD_ANGLE = M_PI / 6;

  SDL_Renderer* renderer;
  SDL_Window* window;
  SDL_Texture* fluid_texture;

  void draw_arrow(uint32_t x,
                  uint32_t y,
                  float length,
                  float angle,
                  float arrow_head_length,
                  float arrow_head_angle);

  void cleanup();

 public:
  GraphicsHandler(uint32_t fluid_cell_size,
                  uint32_t fluid_height,
                  uint32_t fluid_width);
  ~GraphicsHandler();
};