#include <array>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <portableRT/portableRT.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


std::string get_executable_path() {
    char result[1024];
    ssize_t count = readlink("/proc/self/exe", result, 1024);
    return std::string(result, (count > 0) ? count : 0);
}

std::string get_executable_dir() {
    std::string full_path = get_executable_path();
    size_t found = full_path.find_last_of("/\\");
    return full_path.substr(0, found);
}

int main() {

    size_t width = 128;
    size_t height = 128;

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

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    std::vector<std::array<float, 9>> tris{};

    for (size_t s = 0; s < shapes.size(); s++) {

        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if(fv != 3)
                continue;

            std::array<float, 9> tri;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

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
    for(auto backend : portableRT::available_backends()){
        std::cout << i++ <<  " " << backend->name() << std::endl;
    }

    int sel_backend = 0;
    std::cin >> sel_backend;


    portableRT::select_backend(portableRT::available_backends()[sel_backend]);

    float camera_dist = 0.5f;
    float sensor_size = 0.05f;
    float sensor_dist = 0.05f;

    std::vector<unsigned char> image(width * height);

    auto start = std::chrono::high_resolution_clock::now();

    for(int y = 0; y < height; ++y){
        std::cout << "\r" << std::fixed << std::setprecision(1)
          << 100.0f * static_cast<float>(y) / height << "%" << std::flush;
        for(int x = 0; x < width; ++x){

            float sx = sensor_size * (static_cast<float>(x) / width -  0.5);
            float sy = sensor_size * (static_cast<float>(y) / height - 0.5);

            std::array<float,3> camera_pos{0, 0, -camera_dist};
            std::array<float,3> sensor_pos{sx, sy, -camera_dist + sensor_dist};

            portableRT::Ray ray;
            ray.origin = camera_pos;
            ray.direction = {sensor_pos[0] - camera_pos[0], sensor_pos[1] - camera_pos[1], sensor_pos[2] - camera_pos[2]};
            float length = sqrt(sensor_pos[0] * sensor_pos[0] + sensor_pos[1] * sensor_pos[1] + sensor_pos[2] * sensor_pos[2]);
            ray.direction[0] /= length;
            ray.direction[1] /= length;
            ray.direction[2] /= length;

            bool hit = false;

            for(const auto& tri : tris){
                hit |= portableRT::intersect_tri(tri, ray);
            }
            image[width * y + x] = static_cast<unsigned char>(hit * 255.0f);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << static_cast<int>((width * height)/(duration.count()/1000.0f)) << "rays/s: " << std::endl;

    stbi_write_png("bunny_output.png", width, height, 1, image.data(), width);

	return 0;
}