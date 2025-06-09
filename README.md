<p align="center">
  <img
    src="https://github.com/user-attachments/assets/22e38b6f-602a-4f84-b981-f3a243f09457" width="200"style="display:block;margin:-20px auto;" alt="logo"/>
</p>




# portableRT – Portable Ray Tracing 

portableRT is a C++ library that enables your application to perform ray tracing using all available hardware through a single API. On CPUs, it uses Embree. On GPUs, the user can choose between a general GPGPU approach, ideal for older or compute-focused GPUs without dedicated ray tracing units, or make use of the hardware-accelerated ray tracing units (when available)

This first version is extremely simple and focuses solely on intersecting multiples ray with a triangle structure using all the available backends. It currently works with all the available GPUs Ray tracing cores.


## Recommended Backends by Device Type

| Device Type                                      | Recommended Backend      |
|--------------------------------------------------|--------------------------|
| NVIDIA GPUs with RT cores                        | `OptiX`                  |
| AMD GPUs with Ray Accelerators                   | `HIP`                    |
| Intel GPUs with Ray Tracing Units (RTUs)         | `Embree SYCL`            |
| x86_64 CPUs (Intel/AMD)                          | `Embree CPU`             |
| Non-x86 CPUs (e.g. ARM, RISC-V)                  | `CPU`                    |
| GPUs without dedicated ray tracing units (any)   | `SYCL`                   |


## Building

```bash
git clone https://github.com/101001000/portableRT.git
cd portableRT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Optional Back‑Ends

#### **OptiX**

1. [Download OptiX 7.5.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) and extract it.  
2. Configure CMake:

        -DUSE_OPTIX=ON -DOptiX_ROOT=/path/to/OptiX-SDK-7.5.0-linux64-x86_64

---

#### **HIP‑ROCm**

1. [Install HIP with ROCm](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html) (tested with ROCm 5.4.3).  
2. Configure CMake:

        -DUSE_HIP=ON -DHIP_ROOT=/path/to/rocm

---

#### **Embree + SYCL**

1. Grab the SYCL‑enabled build: [embree‑4.4.0.sycl.x86_64.linux.tar.gz](https://github.com/RenderKit/embree/releases/tag/v4.4.0).  
   You can reuse the same install for the CPU back‑end.  
2. Configure CMake:

        -DUSE_EMBREE_SYCL=ON -Dembree_DIR=/path/to/embree-sycl/

---

#### **Embree CPU**

1. Download the CPU build (with or without SYCL): [embree‑4.4.0.x86_64.linux.tar.gz](https://github.com/RenderKit/embree/releases/tag/v4.4.0).  
2. Configure CMake:

        -DUSE_EMBREE_CPU=ON -Dembree_DIR=/path/to/embree/

---

> **Don’t forget**  
> • Source `embree_vars.sh` before using any Embree back‑end.  
> • For SYCL targets (Embree SYCL, generic SYCL), compile with DPC++ 6.0.1 or newer — e.g. [intel/llvm v6.0.1](https://github.com/intel/llvm/tree/v6.0.1).


## Usage


1. [Download the latest release]([https://github.com/101001000/portableRT/releases](https://github.com/101001000/portableRT/releases/tag/v0.1.0)).  
2. Include the headers in your project.  
3. Link against `portableRT.a`.  
4. Make sure the `lib` folder is added to your `LD_LIBRARY_PATH`.

That's it, you're ready to go.


### Selecting a Backend

You can select a backend at runtime using the `select_backend` function, as well as list all the available backends with `available_backends()`.

### Building the acceleration structure

By calling `set_tris(std::vector<std::array<float, 9>>)` and providing an array of triangles, the acceleration structure will be built automatically.

```cpp
backend->set_tris(triangles);
```

### Intersecting rays

By calling `nearest_hits(std::vector<Ray>)` and providing an array of rays, the nearest hit for each ray will be computed and returned as an array of distances to the intersection (or infinity if the ray doesn't intersect any triangle). It can be called as a free function that doesn't relies on virtual dispatch (and the only overhead is just a function pointer indirection), or as a member function of the backend object.

```cpp
std::vector<float> hits = portableRT::nearest_hits({rays});
```

### Example

```cpp
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
    backend->set_tris({vertices});
    std::vector<float> hits1 = portableRT::nearest_hits({hit_ray});
    std::vector<float> hits2 = portableRT::nearest_hits({miss_ray});
    std::cout << "Ray 1: "
              << (hits1[0] == std::numeric_limits<float>::infinity() ? "miss"
                                                                     : "hit")
              << "\nRay 2: "
              << (hits2[0] == std::numeric_limits<float>::infinity() ? "miss"
                                                                     : "hit")
              << std::endl;
  }

  return 0;
}
```