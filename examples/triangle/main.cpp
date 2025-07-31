#include <array>
#include <iostream>
#include <portableRT/portableRT.hpp>

int main() {

	std::array<float, 9> vertices = {-1, -1, 0, 1, -1, 0, 0, 1, 0};

	portableRT::Ray ray;
	ray.origin = std::array<float, 3>{0.1, 0, -1};
	ray.direction = std::array<float, 3>{0, 0, 1};

	for (auto backend : portableRT::available_backends()) {
		std::cout << "Testing " << backend->name() << std::endl;
		portableRT::select_backend(backend);
		backend->set_tris({vertices});
		auto hits = portableRT::nearest_hits({ray});
		std::cout << "Ray valid: " << hits[0].valid << "\nRay t: " << hits[0].t
		          << "\nRay u: " << hits[0].u << "\nRay v: " << hits[0].v
		          << "\nRay primitive_id: " << hits[0].primitive_id << "\nRay px: " << hits[0].px
		          << "\nRay py: " << hits[0].py << "\nRay pz: " << hits[0].pz << std::endl;
	}

	return 0;
}