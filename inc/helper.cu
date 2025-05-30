#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

template <typename T> class Vector2d {
private:
  // the same as i
  T x;
  // the same as j
  T y;

public:
  __device__ __host__ inline Vector2d(T x, T y);
  __device__ __host__ inline Vector2d();

  // getters
  __device__ __host__ inline T get_x() const;
  __device__ __host__ inline T get_y() const;

  // setters
  __device__ __host__ inline void set_x(T x);
  __device__ __host__ inline void set_y(T y);

  __device__ __host__ inline Vector2d<T> operator+(const Vector2d<T> &other);
  __device__ __host__ inline Vector2d<T> operator-(const Vector2d<T> &other);

  template <typename G>
  __device__ __host__ friend Vector2d<T> operator*(const Vector2d<T> &vector,
                                                   G scalar) {
    return Vector2d<T>(vector.x * scalar, vector.y * scalar);
  }
};

template <typename T>
__device__ __host__ inline Vector2d<T>
Vector2d<T>::operator+(const Vector2d<T> &other) {
  return Vector2d<T>(this->x + other.x, this->y + other.y);
}

template <typename T>
__device__ __host__ inline Vector2d<T>
Vector2d<T>::operator-(const Vector2d<T> &other) {
  return Vector2d<T>(this->x - other.x, this->y - other.y);
}

template <typename T>
__device__ __host__ inline Vector2d<T>::Vector2d(T x, T y) : x(x), y(y) {}

template <typename T>
__device__ __host__ inline Vector2d<T>::Vector2d() : x(0), y(0) {}

template <typename T> __device__ __host__ inline T Vector2d<T>::get_x() const {
  return x;
}

template <typename T> __device__ __host__ inline T Vector2d<T>::get_y() const {
  return y;
}

template <typename T> __device__ __host__ inline void Vector2d<T>::set_x(T x) {
  this->x = x;
}

template <typename T> __device__ __host__ inline void Vector2d<T>::set_y(T y) {
  this->y = y;
}

template <typename T>
__device__ __host__ inline float get_distance(Vector2d<T> a, Vector2d<T> b) {
  return std::sqrt((a.get_x() - b.get_x()) * (a.get_x() - b.get_x()) +
                   (a.get_y() - b.get_y()) * (a.get_y() - b.get_y()));
}

__device__ __host__ inline void hsv_to_rgb(float h, float s, float v,
                                           uint8_t &r, uint8_t &g, uint8_t &b) {
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