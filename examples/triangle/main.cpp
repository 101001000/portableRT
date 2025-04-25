#include <portableRT/portableRT.h>
#include <array>
#include <iostream>

int main() {

    std::array<float, 9> vertices = {0,0,0, 1,0,0, 0,1,0};

    portableRT::Ray hit_ray;
    hit_ray.origin = std::array<float, 3>{0,0,-1};
    hit_ray.direction = std::array<float, 3>{0,0,1};

    portableRT::Ray miss_ray;
    miss_ray.origin = std::array<float, 3> {-2,0,-1};
    miss_ray.direction = std::array<float, 3>{0,0,1};

    bool hit1 = portableRT::intersect_tri(vertices, hit_ray);
    bool hit2 = portableRT::intersect_tri(vertices, miss_ray);

    std::cout << "Ray 1: " << hit1 << std::endl;
    std::cout << "Ray 2: " << hit2 << std::endl;  

    return 0;
}