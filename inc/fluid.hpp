#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <stdexcept>

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
  T get_x() const;
  T get_y() const;

  // setters
  void set_x(T x);
  void set_y(T y);
};

template <typename T>
  requires arithmetic_concept<T>
Vector2d<T>::Vector2d(T x, T y) : x(x), y(x) {}

template <typename T>
  requires arithmetic_concept<T>
T Vector2d<T>::get_x() const {
  return x;
}

template <typename T>
  requires arithmetic_concept<T>
T Vector2d<T>::get_y() const {
  return y;
}

template <typename T>
  requires arithmetic_concept<T>
void Vector2d<T>::set_x(T x) {
  this->x = x;
}

template <typename T>
  requires arithmetic_concept<T>
void Vector2d<T>::set_y(T y) {
  this->y = y;
}

class FluidCell {
 private:
  bool is_solid;
  Vector2d<float> velocity;
  float pressure;

 public:
  FluidCell();
  FluidCell(bool is_solid);

  // getters
  bool get_is_solid() const;
  uint8_t get_s() const;
  Vector2d<float> get_velocity() const;
  float get_pressure() const;

  // setters
  void set_velocity_x(float x);
  void set_velocity_y(float y);
  void set_velocity(float x, float y);
};

class SmokeCell {
 private:
  // This value should be between 0 and 1
  float density;

 public:
  SmokeCell();
  float get_density() const;
  void set_density(float density);
};

class Cell {
  FluidCell fluid;
  SmokeCell smoke;

 public:
  Cell();
  Cell(bool is_solid);

  // getters
  const Vector2d<float> get_velocity() const;
  const float get_density() const;
  const bool is_solid() const;
  const uint8_t get_s() const;

  // setters
  void set_velocity_x(float x);
  void set_velocity_y(float y);
  void set_velocity(float x, float y);
  void set_density(float density);
};

template <uint32_t H, uint32_t W>
class Fluid {
 private:
  std::array<std::array<Cell, W>, H> grid;

  Cell& get_mut_cell(uint32_t i, uint32_t j);
  void step_projection(uint32_t i, uint32_t j);

 public:
  const float g = -9.81;
  const float o;
  const uint32_t n;

  Fluid(float o, uint32_t n);

  // getters
  const Cell& get_cell(uint32_t i, uint32_t j) const;
  float get_divergence(uint32_t i, uint32_t j) const;
  uint8_t get_s(uint32_t i, uint32_t j) const;

  bool is_edge(uint32_t i, uint32_t j) const;

  void apply_external_forces(float d_t);
  void perform_projection();
};

template <uint32_t H, uint32_t W>
bool Fluid<H, W>::is_edge(uint32_t i, uint32_t j) const {
  return i == 0 || j == 0 || i == H - 1 || j == W - 1;
}

template <uint32_t H, uint32_t W>
Fluid<H, W>::Fluid(float o, uint32_t n) : o(o), n(n) {
  for (auto i = 0; i < W; i++) {
    for (auto j = 0; j < H; j++) {
      Cell& cell = this->get_mut_cell(i, j);
      cell = Cell(is_edge(i, j));
    }
  }
}

template <uint32_t H, uint32_t W>
const Cell& Fluid<H, W>::get_cell(uint32_t i, uint32_t j) const {
  if (i >= W || j >= H) {
    throw std::out_of_range(std::format(
        "Index out of range while accessing Cell at ({}, {})", i, j));
  }
  return grid[H - j - 1][i];
};

template <uint32_t H, uint32_t W>
Cell& Fluid<H, W>::get_mut_cell(uint32_t i, uint32_t j) {
  if (i >= W || j >= H) {
    throw std::out_of_range(std::format(
        "Index out of range while accessing Cell at ({}, {})", i, j));
  }
  return grid[H - j - 1][i];
};

template <uint32_t H, uint32_t W>
float Fluid<H, W>::get_divergence(uint32_t i, uint32_t j) const {
  const Cell& cell = get_cell(i, j);
  const Cell& top_cell = get_cell(i, j + 1);
  const Cell& right_cell = get_cell(i + 1, j);

  auto u = cell.get_velocity().get_x();
  auto v = cell.get_velocity().get_y();
  auto top_v = top_cell.get_velocity().get_y();
  auto right_u = right_cell.get_velocity().get_x();

  auto divergence = right_u - u + top_v - v;

  return divergence;
}

// ! Possible performance gain by memoizing s for each cell
template <uint32_t H, uint32_t W>
uint8_t Fluid<H, W>::get_s(uint32_t i, uint32_t j) const {
  const Cell& top_cell = get_cell(i, j + 1);
  const Cell& bottom_cell = get_cell(i, j - 1);
  const Cell& right_cell = get_cell(i + 1, j);
  const Cell& left_cell = get_cell(i - 1, j);

  auto top_s = top_cell.get_s();
  auto bottom_s = bottom_cell.get_s();
  auto right_s = right_cell.get_s();
  auto left_s = left_cell.get_s();

  auto s = top_s + bottom_s + right_s + left_s;

  return s;
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::step_projection(uint32_t i, uint32_t j) {
  Cell& cell = get_mut_cell(i, j);
  if (cell.is_solid()) {
    return;
  }

  Cell& top_cell = get_mut_cell(i, j + 1);
  Cell& bottom_cell = get_mut_cell(i, j - 1);
  Cell& right_cell = get_mut_cell(i + 1, j);
  Cell& left_cell = get_mut_cell(i - 1, j);

  auto u = cell.get_velocity().get_x();
  auto v = cell.get_velocity().get_y();
  auto top_v = top_cell.get_velocity().get_y();
  auto right_u = right_cell.get_velocity().get_x();

  auto divergence = get_divergence(i, j);
  auto s = get_s(i, j);
  auto velocity_diff = this->o * (divergence / s);

  if (left_cell.get_s()) {
    u += velocity_diff;
    cell.set_velocity_x(u);
  }

  if (right_cell.get_s()) {
    right_u -= velocity_diff;
    right_cell.set_velocity_x(right_u);
  }

  if (bottom_cell.get_s()) {
    v += velocity_diff;
    cell.set_velocity_y(v);
  }

  if (top_cell.get_s()) {
    top_v -= velocity_diff;
    top_cell.set_velocity_y(top_v);
  }
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::perform_projection() {
  for (uint32_t _ = 0; _ < n; _++) {
    for (uint32_t i = 1; i < H - 1; i++) {
      for (uint32_t j = 1; j < W - 1; j++) {
        step_projection(i, j);
      }
    }
  }
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::apply_external_forces(float d_t) {
  for (uint32_t i = 0; i < H; i++) {
    for (uint32_t j = 0; j < W; j++) {
      Cell& cell = get_mut_cell(i, j);
      if (cell.is_solid()) {
        continue;
      }

      if (j == 10 && i > 5 && i < 11) {
        cell.set_velocity(20, 20);
        cell.set_density(1);
      }
      // auto v = cell.get_velocity().get_y();
      // v += d_t * g;
      // cell.set_velocity_y(v);
    }
  }
}