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

  // getters
  T get_x();
  T get_y();

  // setters
  void set_x(T x);
  void set_y(T y);
};

class FluidCell {
 private:
  bool is_solid;
  Vector2d<double> velocity;
  double pressure;

 public:
  FluidCell(bool is_solid);

  // getters
  bool get_is_solid();
  uint8_t get_s();
  Vector2d<double> get_velocity();
  double get_pressure();
};

class SmokeCell {
 private:
  // This value should be between 0 and 1
  double density;

 public:
  SmokeCell();
};

class Cell {
  FluidCell fluid;
  SmokeCell smoke;

 public:
  Cell(bool is_solid);
  
  // getters
  FluidCell& get_fluid();
  SmokeCell& get_smoke();
};

template <uint32_t H, uint32_t W>
class Fluid {
 private:
  Cell grid[H][W];

 public:
  const double g = -9.81;
  const double o;
  const uint32_t n;

  Fluid(double o, uint32_t n);

  // getters
  Cell& get_cell(uint32_t i, uint32_t j);
  float get_divergence(uint32_t i, uint32_t j);
  uint8_t get_s(uint32_t i, uint32_t j);

  bool is_edge(uint32_t i, uint32_t j);

  void apply_external_forces(double d_t);
  void preform_projection();
};