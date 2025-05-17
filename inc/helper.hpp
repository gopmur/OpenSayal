#pragma once

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <cstdint>

template <typename T>
inline T square(T a) {
  return a * a;
}

template <typename T>
concept arithmetic_concept = std::integral<T> or std::floating_point<T>;

template <typename T>
  requires arithmetic_concept<T>
class Vector2d {
 private:
  // the same as i
  T x;
  // the same as j
  T y;

 public:
  inline Vector2d(T x, T y);
  inline Vector2d();

  // getters
  inline T get_x() const;
  inline T get_y() const;

  // setters
  inline void set_x(T x);
  inline void set_y(T y);

  inline Vector2d<T> operator+(const Vector2d<T>& other);
  inline Vector2d<T> operator-(const Vector2d<T>& other);

  template <typename G>
    requires arithmetic_concept<G>
  friend Vector2d<T> operator*(const Vector2d<T>& vector, G scalar) {
    return Vector2d<T>(vector.x * scalar, vector.y * scalar);
  }

  template <typename G>
    requires arithmetic_concept<G>
  friend Vector2d<T> operator/(const Vector2d<T>& vector, G scalar) {
    return Vector2d<T>(vector.x / scalar, vector.y / scalar);
  }
};

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T> Vector2d<T>::operator+(const Vector2d<T>& other) {
  return Vector2d<T>(this->x + other.x, this->y + other.y);
}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T> Vector2d<T>::operator-(const Vector2d<T>& other) {
  return Vector2d<T>(this->x - other.x, this->y - other.y);
}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T>::Vector2d(T x, T y) : x(x), y(y) {}

template <typename T>
  requires arithmetic_concept<T>
inline Vector2d<T>::Vector2d() : x(0), y(0) {}

template <typename T>
  requires arithmetic_concept<T>
inline T Vector2d<T>::get_x() const {
  return x;
}

template <typename T>
  requires arithmetic_concept<T>
inline T Vector2d<T>::get_y() const {
  return y;
}

template <typename T>
  requires arithmetic_concept<T>
inline void Vector2d<T>::set_x(T x) {
  this->x = x;
}

template <typename T>
  requires arithmetic_concept<T>
inline void Vector2d<T>::set_y(T y) {
  this->y = y;
}

template <typename T>
inline float get_distance(Vector2d<T> a, Vector2d<T> b) {
  return std::sqrt((a.get_x() - b.get_x()) * (a.get_x() - b.get_x()) +
                   (a.get_y() - b.get_y()) * (a.get_y() - b.get_y()));
}

inline void hsv_to_rgb(float h,
                       float s,
                       float v,
                       uint8_t& r,
                       uint8_t& g,
                       uint8_t& b) {
  float c = v * s;
  float x = c * (1 - std::fabs(fmod(h / 60.0f, 2) - 1));
  float m = v - c;

  float r_, g_, b_;
  if (h < 60) {
    r_ = c;
    g_ = x;
    b_ = 0;
  } else if (h < 120) {
    r_ = x;
    g_ = c;
    b_ = 0;
  } else if (h < 180) {
    r_ = 0;
    g_ = c;
    b_ = x;
  } else if (h < 240) {
    r_ = 0;
    g_ = x;
    b_ = c;
  } else if (h < 300) {
    r_ = x;
    g_ = 0;
    b_ = c;
  } else {
    r_ = c;
    g_ = 0;
    b_ = x;
  }

  r = static_cast<uint8_t>((r_ + m) * 255);
  g = static_cast<uint8_t>((g_ + m) * 255);
  b = static_cast<uint8_t>((b_ + m) * 255);
}