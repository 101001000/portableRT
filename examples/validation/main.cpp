#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <portableRT/portableRT.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/util.h"

struct TestResult {
	std::string backend;
	std::string device;
	std::string test;
	std::string sub_test;
	bool result;
	std::string value_str;
	std::string expected_str;

	static constexpr float epsilon = 0.0001f;

	template <typename T> static constexpr T lo(T v) {
		if constexpr (std::is_floating_point_v<T>)
			return v - static_cast<T>(epsilon);
		else
			return v;
	}

	template <typename T> static constexpr T hi(T v) {
		if constexpr (std::is_floating_point_v<T>)
			return v + static_cast<T>(epsilon);
		else
			return v;
	}

	template <typename T>
	TestResult(std::string backend, std::string device, std::string test, std::string sub_test,
	           T value, T min_exp, T max_exp)
	    : backend(std::move(backend)), device(std::move(device)), test(std::move(test)),
	      sub_test(std::move(sub_test)), result(value >= min_exp && value <= max_exp) {
		value_str = std::to_string(value);
		expected_str = std::to_string(min_exp) + " - " + std::to_string(max_exp);
	}

	template <typename T>
	TestResult(std::string backend, std::string device, std::string test, std::string sub_test,
	           T value, T expected)
	    : TestResult(std::move(backend), std::move(device), std::move(test), std::move(sub_test),
	                 value, lo(expected), hi(expected)) {}
};

std::vector<TestResult> results;

void tri_validation(portableRT::Backend *backend) {

	std::array<float, 9> vertices = {-1, -1, 0, 1, -1, 0, 0, 1, 0};

	portableRT::Ray hit_ray;
	hit_ray.origin = std::array<float, 3>{0.1, 0, -1};
	hit_ray.direction = std::array<float, 3>{0, 0, 1};

	portableRT::Ray miss_ray;
	miss_ray.origin = std::array<float, 3>{-2, 0, -1};
	miss_ray.direction = std::array<float, 3>{0, 0, 1};

	portableRT::selected_backend->set_tris({vertices});

	// Full hit validation
	auto hits1_full = portableRT::nearest_hits({hit_ray});
	auto hits2_full = portableRT::nearest_hits({miss_ray});

	// TODO: añadir p
	results.push_back(TestResult("FullReg Hit", "valid", backend->name(), backend->device_name(),
	                             hits1_full[0].valid, hits1_full[0].valid));
	results.push_back(TestResult("FullReg Hit", "t", backend->name(), backend->device_name(),
	                             hits1_full[0].t, 1.0f));
	results.push_back(TestResult("FullReg Hit", "u", backend->name(), backend->device_name(),
	                             hits1_full[0].u, 0.3f));
	results.push_back(TestResult("FullReg Hit", "v", backend->name(), backend->device_name(),
	                             hits1_full[0].v, 0.5f));
	results.push_back(TestResult("FullReg Hit", "primitive_id", backend->name(),
	                             backend->device_name(), hits1_full[0].primitive_id, 0u));
	results.push_back(TestResult("FullReg Hit", "px", backend->name(), backend->device_name(),
	                             hits1_full[0].px, 0.1f));
	results.push_back(TestResult("FullReg Hit", "py", backend->name(), backend->device_name(),
	                             hits1_full[0].py, 0.0f));
	results.push_back(TestResult("FullReg Hit", "pz", backend->name(), backend->device_name(),
	                             hits1_full[0].pz, 0.0f));
	results.push_back(TestResult("FullReg Miss", "valid", backend->name(), backend->device_name(),
	                             hits2_full[0].valid, false));

	// Filtered hit validation
	auto hits1_filtered = portableRT::nearest_hits<portableRT::filter::valid>({hit_ray});
	auto hits2_filtered = portableRT::nearest_hits<portableRT::filter::valid>({miss_ray});

	results.push_back(TestResult("Filtered Hit", "valid", backend->name(), backend->device_name(),
	                             hits1_filtered[0].valid, true));
	results.push_back(TestResult("Filtered Miss", "valid", backend->name(), backend->device_name(),
	                             hits2_filtered[0].valid, false));
}

std::pair<std::vector<std::array<float, 9>>, std::vector<portableRT::Ray>> initialize_bunny() {
	constexpr size_t width = 1024;
	constexpr size_t height = 1024;

	std::string inputfile = get_executable_dir() + "/common/bunny.obj";
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(inputfile, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto &attrib = reader.GetAttrib();
	auto &shapes = reader.GetShapes();
	auto &materials = reader.GetMaterials();

	std::vector<std::array<float, 9>> tris{};

	for (size_t s = 0; s < shapes.size(); s++) {

		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
			if (fv != 3)
				continue;

			std::array<float, 9> tri;

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				tri[v * 3 + 0] = vx;
				tri[v * 3 + 1] = vy;
				tri[v * 3 + 2] = vz;
			}

			tris.push_back(tri);

			index_offset += fv;
		}
	}

	std::vector<portableRT::Ray> rays;

	float camera_dist = 0.5f;
	float sensor_size = 0.05f;
	float sensor_dist = 0.05f;

	for (int y = height - 1; y >= 0; --y) {
		for (int x = 0; x < width; ++x) {

			float sx = sensor_size * (static_cast<float>(x) / width - 0.5);
			float sy = sensor_size * (static_cast<float>(y) / height - 0.5);

			std::array<float, 3> camera_pos{0, 0, -camera_dist};
			std::array<float, 3> sensor_pos{sx, sy, -camera_dist + sensor_dist};

			portableRT::Ray ray;
			ray.origin = camera_pos;
			ray.direction = {sensor_pos[0] - camera_pos[0], sensor_pos[1] - camera_pos[1],
			                 sensor_pos[2] - camera_pos[2]};
			float length = sqrt(sensor_pos[0] * sensor_pos[0] + sensor_pos[1] * sensor_pos[1] +
			                    sensor_pos[2] * sensor_pos[2]);

			ray.direction[0] /= length;
			ray.direction[1] /= length;
			ray.direction[2] /= length;

			rays.push_back(ray);
		}
	}

	return {tris, rays};
}

static auto [tris, rays] = initialize_bunny();
static int v_width, v_height, v_channels;
static unsigned char *v_data = stbi_load((get_executable_dir() + "/common/bunny.png").c_str(),
                                         &v_width, &v_height, &v_channels, 0);

void bunny_validation(portableRT::Backend *backend) {

	auto bvh_start = std::chrono::high_resolution_clock::now();
	backend->set_tris(tris);
	auto bvh_end = std::chrono::high_resolution_clock::now();
	auto bvh_duration = std::chrono::duration_cast<std::chrono::microseconds>(bvh_end - bvh_start);

	auto traverse_start = std::chrono::high_resolution_clock::now();
	auto hits = portableRT::nearest_hits(rays);
	auto traverse_end = std::chrono::high_resolution_clock::now();
	auto traverse_duration =
	    std::chrono::duration_cast<std::chrono::microseconds>(traverse_end - traverse_start);

	bool pixel_validation = true;
	for (size_t i = 0; i < hits.size(); i++) {
		pixel_validation &= (hits[i].valid ? 255 : 0) == v_data[i];
	}

	results.push_back(TestResult("Bunny", "Pixel Validation", backend->name(),
	                             backend->device_name(), pixel_validation, true));
	results.push_back(TestResult("Bunny", "BVH Build time", backend->name(), backend->device_name(),
	                             bvh_duration.count() / 1000.0f, 0.0f, 1000.0f));
	results.push_back(TestResult("Bunny", "Traverse time", backend->name(), backend->device_name(),
	                             traverse_duration.count() / 1000.0f, 0.0f, 1000.0f));
}

int main() {

	for (auto backend : portableRT::available_backends()) {

		std::cout << "Testing " << backend->name() << std::endl;

		portableRT::select_backend(backend);
		tri_validation(portableRT::selected_backend);
		bunny_validation(portableRT::selected_backend);
	}

	stbi_image_free(v_data);

	std::ofstream file("results.csv");
	file << "Backend,Device,Test,Subtest,Value,Expected,Validation\n";

	for (const auto &result : results) {
		if (!result.result) {
			std::cout << "Validation failed in test: " << result.backend << "," << result.device
			          << "," << result.test << "," << result.sub_test << "," << result.value_str
			          << "," << result.expected_str << "," << result.result << "\n";
		}
		file << result.backend << "," << result.device << "," << result.test << ","
		     << result.sub_test << "," << result.value_str << "," << result.expected_str << ","
		     << result.result << "\n";
	}

	return 0;
}