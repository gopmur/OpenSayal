#include "fluid.hpp"

template <typename T>
  requires arithmetic_concept<T>
Vector2d<T>::Vector2d(T x, T y) : x(x), y(x) {}

FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

SmokeCell::SmokeCell() : density(0) {}

Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}

template <uint32_t H, uint32_t W>
bool Fluid<H, W>::is_edge(uint32_t i, uint32_t j) {
  return i == 0 || j == 0 || i == H - 1 || j == W - 1;
}

template <uint32_t H, uint32_t W>
Fluid<H, W>::Fluid() {
  for (auto i = 0; i < H; i++) {
    for (auto j = 0; j < W; j++) {
      grid[i][j] = Cell(is_edge(i, j));
    }
  }
}