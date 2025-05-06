#include <portableRT/portableRT.h>
#include <array>
#include <iostream>

int main() {

    std::array<float, 9> vertices = {-1,-1,0, 1,-1,0, 0,1,0};

    portableRT::Ray hit_ray;
    hit_ray.origin = std::array<float, 3>{0,0,-1};
    hit_ray.direction = std::array<float, 3>{0,0,1};

    portableRT::Ray miss_ray;
    miss_ray.origin = std::array<float, 3> {-2,0,-1};
    miss_ray.direction = std::array<float, 3>{0,0,1};

    std::cout << "Testing CPU" << std::endl;{
    bool hit1 = portableRT::intersect_tri<portableRT::Backend::CPU>(vertices, hit_ray);
    bool hit2 = portableRT::intersect_tri<portableRT::Backend::CPU>(vertices, miss_ray);
    std::cout << "Ray 1: " << hit1 << "\nRay 2: " << hit2 << std::endl;}

#ifdef USE_OPTIX
    std::cout << "Testing OPTIX" << std::endl;{
    bool hit1 = portableRT::intersect_tri<portableRT::Backend::OPTIX>(vertices, hit_ray);
    bool hit2 = portableRT::intersect_tri<portableRT::Backend::OPTIX>(vertices, miss_ray);
    std::cout << "Ray 1: " << hit1 << "\nRay 2: " << hit2 << std::endl;}
#endif

#ifdef USE_HIP
    std::cout << "Testing HIP" << std::endl;{
    bool hit1 = portableRT::intersect_tri<portableRT::Backend::HIP>(vertices, hit_ray);
    bool hit2 = portableRT::intersect_tri<portableRT::Backend::HIP>(vertices, miss_ray);
    std::cout << "Ray 1: " << hit1 << "\nRay 2: " << hit2 << std::endl;}
#endif

    return 0;
}