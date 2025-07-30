#include <array>
#include <iostream>
#include <portableRT/portableRT.hpp>

int main() {

	std::array<float, 9> vertices = {-1, -1, 0, 1, -1, 0, 0, 1, 0};

	portableRT::Ray hit_ray;
	hit_ray.origin = std::array<float, 3>{0.1, 0, -1};
	hit_ray.direction = std::array<float, 3>{0, 0, 1};

	portableRT::Ray miss_ray;
	miss_ray.origin = std::array<float, 3>{-2, 0, -1};
	miss_ray.direction = std::array<float, 3>{0, 0, 1};

	for (auto backend : portableRT::available_backends()) {
		std::cout << "Testing " << backend->name() << std::endl;
		portableRT::select_backend(backend);
		backend->set_tris({vertices});
		auto hits1 = portableRT::nearest_hits({hit_ray});
		auto hits2 = portableRT::nearest_hits({miss_ray});
		std::cout << "Ray 1: " << hits1[0].valid << "\nRay 2: " << hits2[0].valid << std::endl;
	}

	return 0;
}