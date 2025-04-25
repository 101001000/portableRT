#include <array>

namespace portableRT {
    struct Ray {
        std::array<float, 3> origin;
        std::array<float, 3> direction;
    };
    bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
}