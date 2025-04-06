#include "fluid.hpp"
#include <stdint.h>
#include <format>
#include <stdexcept>

// -------------------------------- Vector2d -------------------------------- //

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

// -------------------------------- FluidCell ------------------------------- //

FluidCell::FluidCell() : velocity(0, 0), is_solid(0), pressure(0) {}

FluidCell::FluidCell(bool is_solid)
    : velocity(0, 0), is_solid(is_solid), pressure(0) {}

bool FluidCell::get_is_solid() const {
  return is_solid;
}

Vector2d<float> FluidCell::get_velocity() const {
  return velocity;
}

float FluidCell::get_pressure() const {
  return pressure;
}

uint8_t FluidCell::get_s() const {
  return !is_solid;
}

// -------------------------------- SmokeCell ------------------------------- //

SmokeCell::SmokeCell() : density(0) {}

float SmokeCell::get_density() const {
  return this->density;
}

// ---------------------------------- Cell ---------------------------------- //

Cell::Cell(bool is_solid) : smoke(), fluid(is_solid) {}
Cell::Cell() : smoke(), fluid() {}

const FluidCell& Cell::get_fluid() const {
  return fluid;
}

const SmokeCell& Cell::get_smoke() const {
  return smoke;
}
