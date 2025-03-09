#include <concepts>

template <typename T>
concept arithmetic_concept = std::integral<T> or std::floating_point<T>;
