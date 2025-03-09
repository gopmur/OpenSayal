#pragma once

#include <cstdint>

#include "helper.hpp"

template <typename T>
  requires arithmetic_concept<T>
class Vector2d {
 private:
  // the same as i
  T x;
  // the same as j
  T y;

 public:
  Vector2d(T x, T y);
};

class FluidCell {
 private:
  bool is_solid;
  Vector2d<double> velocity;
  double pressure;

 public:
  FluidCell(bool is_solid);
};

class SmokeCell {
 private:
  // This value should be between 0 and 1
  double density;

 public:
  SmokeCell();
};

class Cell {
 private:
  SmokeCell smoke;
  FluidCell fluid;

 public:
  Cell(bool is_solid);
};

template <uint32_t H, uint32_t W>
class Fluid {
 private:
  Cell grid[H][W];

 public:
  Fluid();
  bool is_edge(uint32_t i, uint32_t j);
};