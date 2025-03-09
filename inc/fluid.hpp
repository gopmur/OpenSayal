#include <concepts>
#include <cstdint>

template <typename T>
concept arithmetic = std::integral<T> or std::floating_point<T>;

template <typename T>
  requires arithmetic<T>
class vector_2d {
  // the same as i
  T x;
  // the same as j
  T y;
};

class FluidCell {
  bool is_solid;
  vector_2d<double> velocity;
  double pressure;
};

class SmokeCell {
  // This value should be between 0 and 1
  double density;
};

class Cell {
  SmokeCell smoke;
  FluidCell fluid;
};

template <uint32_t H, uint32_t W>
class Fluid {
  Cell grid[H][W];
};