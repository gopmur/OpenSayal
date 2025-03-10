#include "fluid.hpp"
#include <format>
#include <stdexcept>

// -------------------------------- Vector2d -------------------------------- //

template <typename T>
  requires arithmetic_concept<T>
Vector2d<T>::Vector2d(T x, T y) : x(x), y(x) {}

template <typename T>
  requires arithmetic_concept<T>
T Vector2d<T>::get_x() {
  return x;
}

template <typename T>
  requires arithmetic_concept<T>
T Vector2d<T>::get_y() {
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

// -------------------------------- FluidCell ------------------------------- //

FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

bool FluidCell::get_is_solid() {
  return is_solid;
}

Vector2d<double> FluidCell::get_velocity() {
  return velocity;
}

double FluidCell::get_pressure() {
  return pressure;
}

uint8_t FluidCell::get_s() {
  return !is_solid;
}

// -------------------------------- SmokeCell ------------------------------- //

SmokeCell::SmokeCell() : density(0) {}

// ---------------------------------- Cell ---------------------------------- //

Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}

FluidCell& Cell::get_fluid() {
  return fluid;
}

SmokeCell& Cell::get_smoke() {
  return smoke;
}

// --------------------------------- Fluid ---------------------------------- //

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

template <uint32_t H, uint32_t W>
Cell& Fluid<H, W>::get_cell(uint32_t i, uint32_t j) {
  if (i >= H || j >= W) {
    throw std::out_of_range(std::format(
        "Index out of range while accessing Cell at ({}, {})", i, j));
  }
  return grid[i][j];
};

template <uint32_t H, uint32_t W>
float Fluid<H, W>::get_divergence(uint32_t i, uint32_t j) {
  Cell& cell = get_cell(i, j);
  Cell& top_cell = get_cell(i - 1, j);
  Cell& right_cell = get_cell(i, j + 1);

  auto u = cell.get_fluid().get_velocity().get_x();
  auto v = cell.get_fluid().get_velocity().get_y();
  auto top_v = top_cell.get_fluid().get_velocity().get_y();
  auto right_u = right_cell.get_fluid().get_velocity().get_x();

  auto divergence = right_u - u + top_v - v;

  return divergence;
}

// ! Possible performance gain by memoizing s for each cell
template <uint32_t H, uint32_t W>
uint8_t Fluid<H, W>::get_s(uint32_t i, uint32_t j) {
  Cell& top_cell = get_cell(i - 1, j);
  Cell& bottom_cell = get_cell(i + 1, j);
  Cell& right_cell = get_cell(i, j + 1);
  Cell& left_cell = get_cell(i, j - 1);

  auto top_s = top_cell.get_fluid().get_s();
  auto bottom_s = bottom_cell.get_fluid().get_s();
  auto right_s = right_cell.get_fluid().get_s();
  auto left_s = left_cell.get_fluid().get_s();  

  auto s = top_s + bottom_s + right_s + left_s;

  return s;
}

template <uint32_t H, uint32_t W>
void Fluid<H, W>::preform_projection(uint32_t n) {
  for (uint32_t _ = 0; _ < n; _++) {
    for (uint32_t i = 1; i < H - 1; i++) {
      for (uint32_t j = 1; j < W - 1; j++) {
        Cell &cell = get_cell(i, j);
        if (cell.get_fluid().get_is_solid()) {
          continue;
        }

        Cell &top_cell = get_cell(i - 1, j);
        Cell &bottom_cell = get_cell(i + 1, j);
        Cell &right_cell = get_cell(i, j + 1);
        Cell &left_cell = get_cell(i, j - 1);

        auto u = cell.get_fluid().get_velocity().get_x();
        auto v = cell.get_fluid().get_velocity().get_y();
        auto top_v = top_cell.get_fluid().get_velocity().get_y();
        auto right_u = right_cell.get_fluid().get_velocity().get_x();

        auto divergence = get_divergence(i, j);
        auto s = get_s(i, j);
        auto velocity_diff = divergence / s;

        if (left_cell.get_fluid().get_s()) {
          u += velocity_diff;
        }

        if (right_cell.get_fluid().get_s()) {
          right_u -= velocity_diff;
        }

        if (bottom_cell.get_fluid().get_s()) {
          v += velocity_diff;
        }

        if (top_cell.get_fluid().get_s()) {
          top_v -= velocity_diff;
        }
        
      } 
    }
  }
}
