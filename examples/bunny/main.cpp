#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <portableRT/portableRT.hpp>
#include <string>
#include <unistd.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "../common/util.h"

int main() {

	constexpr size_t width = 4096;
	constexpr size_t height = 4096;

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

	std::cout << "Select backend: " << std::endl;
	int i = 0;
	for (auto backend : portableRT::available_backends()) {
		std::cout << i++ << " " << backend->name() << std::endl;
	}

	int sel_backend = 0;
	std::cin >> sel_backend;

	portableRT::select_backend(portableRT::available_backends()[sel_backend]);

	float camera_dist = 0.3f;
	float sensor_size = 0.05f;
	float sensor_dist = 0.05f;

	std::vector<unsigned char> image(width * height);

	auto bvh_start = std::chrono::high_resolution_clock::now();
	portableRT::selected_backend->set_tris(tris);
	auto bvh_end = std::chrono::high_resolution_clock::now();
	auto bvh_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bvh_end - bvh_start);

	std::cout << "BVH building time: " << bvh_duration.count() << " ms" << std::endl;

	std::vector<portableRT::Ray> rays;

	for (int y = height - 1; y >= 0; --y) {
		for (int x = 0; x < width; ++x) {

			float sx = sensor_size * (static_cast<float>(x) / width - 0.5);
			float sy = sensor_size * (static_cast<float>(y) / height - 0.5);

			std::array<float, 3> camera_pos{0, -0.02, -camera_dist};
			std::array<float, 3> sensor_pos{sx, sy, -camera_dist + sensor_dist};

			portableRT::Ray ray;
			ray.origin = camera_pos;
			ray.direction = {sensor_pos[0] - camera_pos[0], sensor_pos[1] - camera_pos[1],
			                 sensor_pos[2] - camera_pos[2]};
			float length =
			    sqrt(ray.direction[0] * ray.direction[0] + ray.direction[1] * ray.direction[1] +
			         ray.direction[2] * ray.direction[2]);

			ray.direction[0] /= length;
			ray.direction[1] /= length;
			ray.direction[2] /= length;

			rays.push_back(ray);
		}
	}

	auto start = std::chrono::high_resolution_clock::now();
	auto hits = portableRT::nearest_hits(rays);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
	std::cout << static_cast<int>((width * height) / (duration.count() / 1000.0f)) << "rays/s"
	          << std::endl;

	for (size_t i = 0; i < hits.size(); i++) {
		image[i] = hits[i].valid ? 255 : 0;
	}

	stbi_write_png("bunny_output.png", width, height, 1, image.data(), width);

	return 0;
}