#pragma once

#include <cstdint>

template <typename T>
class Vector2d {
 private:
  T x;
  T y;

 public:
  __device__ __host__ Vector2d(T x, T y);
  __device__ __host__ Vector2d();

  __device__ __host__ T get_x() const;
  __device__ __host__ T get_y() const;

  __device__ __host__ void set_x(T x);
  __device__ __host__ void set_y(T y);

  __device__ __host__ Vector2d<T> operator+(const Vector2d<T>& other);
  __device__ __host__ Vector2d<T> operator-(const Vector2d<T>& other);

  template <typename G>
  __device__ __host__ friend Vector2d<T> operator*(const Vector2d<T>& vector,
                                                   G scalar) {
    return Vector2d<T>(vector.x * scalar, vector.y * scalar);
  }

  template <typename G>
  __device__ __host__ friend Vector2d<T> operator/(const Vector2d<T>& vector,
                                                   G scalar) {
    return Vector2d<T>(vector.x / scalar, vector.y / scalar);
  }
};

template <typename T>
__device__ __host__ Vector2d<T> Vector2d<T>::operator+(
    const Vector2d<T>& other) {
  return Vector2d<T>(this->x + other.x, this->y + other.y);
}

template <typename T>
__device__ __host__ Vector2d<T> Vector2d<T>::operator-(
    const Vector2d<T>& other) {
  return Vector2d<T>(this->x - other.x, this->y - other.y);
}

template <typename T>
__device__ __host__ Vector2d<T>::Vector2d(T x, T y) : x(x), y(y) {}

template <typename T>
__device__ __host__ Vector2d<T>::Vector2d() : x(0), y(0) {}

template <typename T>
__device__ __host__ T Vector2d<T>::get_x() const {
  return x;
}

template <typename T>
__device__ __host__ T Vector2d<T>::get_y() const {
  return y;
}

template <typename T>
__device__ __host__ void Vector2d<T>::set_x(T x) {
  this->x = x;
}

template <typename T>
__device__ __host__ void Vector2d<T>::set_y(T y) {
  this->y = y;
}

template <typename T>
__device__ __host__ float get_distance(Vector2d<T> a, Vector2d<T> b) {
  return std::sqrt((a.get_x() - b.get_x()) * (a.get_x() - b.get_x()) +
                   (a.get_y() - b.get_y()) * (a.get_y() - b.get_y()));
}

__device__ __host__ int map_rgba(int r, int g, int b, int a);
__device__ __host__ void hsv_to_rgb(float h,
                                    float s,
                                    float v,
                                    uint8_t& r,
                                    uint8_t& g,
                                    uint8_t& b);

template <typename T>
__device__ __host__ inline T square(T a) {
  return a * a;
}

template <typename T>
__host__ __device__ T clamp(T x, T lower, T upper) {
  return (x < lower) ? lower : (x > upper) ? upper : x;
}

template <typename T>
__host__ __device__ bool is_in_radius(T x, T y, T c_x, T c_y, T r) {
  return square(x - c_x) + square(y - c_y) < square(r);
}