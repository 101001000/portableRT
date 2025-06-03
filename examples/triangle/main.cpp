#include <array>
#include <iostream>
#include <portableRT/portableRT.hpp>

int main() {

  std::array<float, 9> vertices = {-1, -1, 0, 1, -1, 0, 0, 1, 0};

  portableRT::Ray hit_ray;
  hit_ray.origin = std::array<float, 3>{0, 0, -1};
  hit_ray.direction = std::array<float, 3>{0, 0, 1};

  portableRT::Ray miss_ray;
  miss_ray.origin = std::array<float, 3>{-2, 0, -1};
  miss_ray.direction = std::array<float, 3>{0, 0, 1};

  for (auto backend : portableRT::available_backends()) {
    std::cout << "Testing " << backend->name() << std::endl;
    portableRT::select_backend(backend);
    bool hit1 = portableRT::intersect_tris({vertices}, hit_ray);
    bool hit2 = portableRT::intersect_tris({vertices}, miss_ray);
    std::cout << "Ray 1: " << hit1 << "\nRay 2: " << hit2 << std::endl;
  }

  return 0;
}