#include <SDL2/SDL.h>
#include <chrono>
#include <iomanip>
#include <portableRT/portableRT.hpp>
#include <unordered_map>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "../common/util.h"

struct Texture {
	int w, h;
	std::vector<uint8_t> pix;
};
std::vector<Texture> textures;
std::vector<cgltf_material> materials;
std::vector<int> mat_ids; // Tri - Material ID
std::vector<std::array<float, 6>> uvs;

std::unordered_map<const cgltf_material *, int> mat_id;

int main(int argc, char **argv) {
	cgltf_options options = {};
	cgltf_data *data = NULL;
	cgltf_result result = cgltf_parse_file(
	    &options, (get_executable_dir() + std::string("/assets/Sponza/Sponza.gltf")).c_str(),
	    &data);
	cgltf_load_buffers(&options, data,
	                   (get_executable_dir() + "/assets/Sponza/Sponza.gltf").c_str());

	std::vector<std::array<float, 9>> tris;

	if (result == cgltf_result_success) {

		for (size_t i = 0; i < data->materials_count; ++i) {
			const cgltf_material *m = &data->materials[i];
			const cgltf_texture_view *tv = &m->pbr_metallic_roughness.base_color_texture;
			if (!tv || !tv->texture || !tv->texture->image)
				continue;

			const cgltf_image *img = tv->texture->image;
			int w, h, c;
			unsigned char *p = stbi_load(
			    (get_executable_dir() + "/assets/Sponza/" + img->uri).c_str(), &w, &h, &c, 4);
			if (!p) {
				std::cerr << "Error cargando textura: " << img->uri << "\n";
				continue;
			}
			textures.push_back({w, h, std::vector<uint8_t>(p, p + w * h * 4)});
			stbi_image_free(p);
			materials.push_back(*m);
			mat_id[m] = int(materials.size()) - 1;
		}

		for (size_t i = 0; i < data->meshes_count; ++i) {
			const cgltf_mesh &mesh = data->meshes[i];

			for (size_t p = 0; p < mesh.primitives_count; ++p) {
				const cgltf_primitive &prim = mesh.primitives[p];
				if (prim.type != cgltf_primitive_type_triangles)
					continue;

				const cgltf_accessor *pos_accessor = nullptr;
				const cgltf_accessor *uv_accessor = nullptr;
				const cgltf_accessor *idx_accessor = prim.indices;

				for (size_t a = 0; a < prim.attributes_count; ++a) {
					if (prim.attributes[a].type == cgltf_attribute_type_position) {
						pos_accessor = prim.attributes[a].data;
					}
					if (prim.attributes[a].type == cgltf_attribute_type_texcoord) {
						uv_accessor = prim.attributes[a].data;
					}
				}

				if (!pos_accessor || !idx_accessor || !uv_accessor)
					continue;

				float *positions =
				    (float *)(((uint8_t *)pos_accessor->buffer_view->buffer->data) +
				              pos_accessor->buffer_view->offset + pos_accessor->offset);

				void *indices_data = ((uint8_t *)idx_accessor->buffer_view->buffer->data) +
				                     idx_accessor->buffer_view->offset + idx_accessor->offset;

				float *uvs_data = (float *)((uint8_t *)uv_accessor->buffer_view->buffer->data +
				                            uv_accessor->buffer_view->offset + uv_accessor->offset);

				auto get_index = [&](int i) -> int {
					switch (idx_accessor->component_type) {
					case cgltf_component_type_r_8u:
						return ((uint8_t *)indices_data)[i];
					case cgltf_component_type_r_16u:
						return ((uint16_t *)indices_data)[i];
					case cgltf_component_type_r_32u:
						return ((uint32_t *)indices_data)[i];
					default:
						return 0;
					}
				};

				for (size_t i = 0; i + 2 < idx_accessor->count; i += 3) {
					std::array<float, 9> tri;
					std::array<float, 6> uv;
					for (int j = 0; j < 3; ++j) {
						int idx = get_index(i + j);
						tri[j * 3 + 0] = positions[idx * 3 + 0];
						tri[j * 3 + 1] = positions[idx * 3 + 1];
						tri[j * 3 + 2] = positions[idx * 3 + 2];
						uv[j * 2 + 0] = uvs_data[idx * 2 + 0];
						uv[j * 2 + 1] = uvs_data[idx * 2 + 1];
					}
					mat_ids.push_back(mat_id[prim.material]);
					tris.push_back(tri);
					uvs.push_back(uv);
				}
			}
		}
		cgltf_free(data);
	}

	constexpr size_t width = 512;
	constexpr size_t height = 512;

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

	uint8_t *pixels = new uint8_t[width * height * 4];

	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *win =
	    SDL_CreateWindow("RT", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
	SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
	SDL_Texture *tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING,
	                                     width, height);

	std::array<float, 3> camera_pos{288, 150, -50};
	float camera_vel = 10.0f;
	float camera_angle = -1.58;
	float camera_wvel = 0.05f;

	bool running = true;
	while (running) {

		auto frame_start = std::chrono::high_resolution_clock::now();

		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_QUIT) {
				running = false;
			} else if (e.type == SDL_KEYDOWN) {
				switch (e.key.keysym.sym) {
				case SDLK_w:
					camera_pos[2] += camera_vel * cos(camera_angle);
					camera_pos[0] += camera_vel * sin(camera_angle);
					break;
				case SDLK_s:
					camera_pos[2] -= camera_vel * cos(camera_angle);
					camera_pos[0] -= camera_vel * sin(camera_angle);
					break;
				case SDLK_d:
					camera_pos[0] += camera_vel * cos(camera_angle);
					camera_pos[2] -= camera_vel * sin(camera_angle);
					break;
				case SDLK_a:
					camera_pos[0] -= camera_vel * cos(camera_angle);
					camera_pos[2] += camera_vel * sin(camera_angle);
					break;
				case SDLK_SPACE:
					camera_pos[1] += camera_vel;
					break;
				case SDLK_LSHIFT:
					camera_pos[1] -= camera_vel;
					break;
				case SDLK_RIGHT:
					camera_angle += camera_wvel;
					break;
				case SDLK_LEFT:
					camera_angle -= camera_wvel;
					break;
				default:
					break;
				}
			}
		}

		rays.clear();

		for (int y = height - 1; y >= 0; --y) {
			for (int x = 0; x < width; ++x) {

				float sx = sensor_size * (static_cast<float>(x) / width - 0.5);
				float sy = sensor_size * (static_cast<float>(y) / height - 0.5);

				float ca = std::cos(camera_angle);
				float sa = std::sin(camera_angle);

				float dx = ca * sx + sa * sensor_dist;
				float dz = -sa * sx + ca * sensor_dist;

				std::array<float, 3> sensor_pos{camera_pos[0] + dx, camera_pos[1] + sy,
				                                camera_pos[2] + dz};

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

		auto rt_start = std::chrono::high_resolution_clock::now();
		auto hits = portableRT::nearest_hits(rays);
		auto rt_end = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < hits.size(); i++) {
			bool hit = hits[i].t != std::numeric_limits<float>::infinity();
			int primitive_id = hits[i].primitive_id;
			int material_id = mat_ids[primitive_id];
			if (!hit) {
				pixels[i * 4 + 0] = 0;
				pixels[i * 4 + 1] = 0;
				pixels[i * 4 + 2] = 0;
				pixels[i * 4 + 3] = 255;
				continue;
			} else {

				float t0 = hits[i].u;
				float t1 = hits[i].v;
				float t2 = 1 - hits[i].u - hits[i].v;

				float u = t2 * uvs[primitive_id][0] + t0 * uvs[primitive_id][2] +
				          t1 * uvs[primitive_id][4];
				float v = t2 * uvs[primitive_id][1] + t0 * uvs[primitive_id][3] +
				          t1 * uvs[primitive_id][5];

				u = u - std::floor(u);
				v = v - std::floor(v);

				const Texture &tex = textures[material_id];

				int x = std::clamp(int(u * tex.w), 0, tex.w - 1);
				int y = std::clamp(int(v * tex.h), 0, tex.h - 1);

				int idx = (y * tex.w + x) * 4;

				pixels[i * 4 + 0] = 1 * tex.pix[idx + 0];
				pixels[i * 4 + 1] = 1 * tex.pix[idx + 1];
				pixels[i * 4 + 2] = 1 * tex.pix[idx + 2];
				pixels[i * 4 + 3] = 255;
			}
		}

		SDL_UpdateTexture(tex, nullptr, pixels, width * 4);
		SDL_RenderClear(ren);
		SDL_RenderCopy(ren, tex, nullptr, nullptr);
		SDL_RenderPresent(ren);

		auto frame_end = std::chrono::high_resolution_clock::now();
		auto frame_duration =
		    std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
		auto rt_duration = std::chrono::duration_cast<std::chrono::microseconds>(rt_end - rt_start);

		std::cout << std::fixed << std::setprecision(3) << "\r Position: " << camera_pos[0] << ", "
		          << camera_pos[1] << ", " << camera_pos[2] << " | "
		          << "Frame time: " << frame_duration.count() / 1000.0 << " ms | "
		          << "RT time: " << rt_duration.count() / 1000.0 << " ms ("
		          << rt_duration.count() / static_cast<float>(frame_duration.count()) * 100 << "%)"
		          << std::flush;
	}

	SDL_DestroyTexture(tex);
	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	delete[] pixels;
}