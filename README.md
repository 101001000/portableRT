# portableRT – Portable Ray Tracing

portableRT is a C++ library that enables your application to perform ray tracing using all available hardware through a single API. On CPUs, it uses Embree. On GPUs, the user can choose between a general GPGPU approach, ideal for older or compute-focused GPUs without dedicated ray tracing units, or make use of the hardware-accelerated ray tracing units (when available)

This first version is extremely simple and focuses solely on intersecting a ray with a triangle using all the available backends. Scene configuration, parallel execution, automatic backend selection, ray scheduling and many more features are planned for future releases.


## Backend Support

portableRT is being developed with multiple backends to support a wide range of devices. The current backends and their development status are:

| Backend        | Status         |
|----------------|----------------|
| `CPU_SCALAR`   | ✅ Works        |
| `OPTIX`        | ✅ Works        |
| `HIP_ROCM`     | ✅ Works        |
| `EMBREE_CPU`   | ✅ Works        |
| `EMBREE_SYCL`  | ✅ Works        |
| `SYCL`         | ✅ Works        |

## Recommended Backends by Device Type

| Device Type                                      | Recommended Backend      |
|--------------------------------------------------|--------------------------|
| NVIDIA GPUs with RT cores                        | `OPTIX`                  |
| AMD GPUs with Ray Accelerators                   | `HIP_ROCM`               |
| Intel GPUs with Ray Tracing Units (RTUs)         | `EMBREE_SYCL`            |
| x86_64 CPUs (Intel/AMD)                          | `EMBREE_CPU`             |
| Non-x86 CPUs (e.g. ARM, RISC-V)                  | `CPU_SCALAR`             |
| GPUs without dedicated ray tracing units (any)   | `SYCL`                   |


## Building

```bash
git clone https://github.com/101001000/portableRT.git
cd portableRT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Optional backends

To enable **OptiX**, add:

```bash
-DUSE_OPTIX=ON -DOptiX_ROOT=/path/to/OptiX-SDK-7.5.0-linux64-x86_64
```

To enable **HIP-ROCM**, add:

```bash
-DUSE_HIP=ON -DHIP_ROOT=/path/to/rocm
```

## Example

```cpp
#include <portableRT/portableRT.h>
#include <array>
#include <iostream>

int main() {
    std::array<float, 9> tri = {-1,-1,0, 1,-1,0, 0,1,0};

    portableRT::Ray hit{{0,0,-1}, {0,0,1}};
    portableRT::Ray miss{{-2,0,-1}, {0,0,1}};

    bool hit_cpu = portableRT::intersect_tri<portableRT::Backend::CPU>(tri, hit);
    bool miss_cpu = portableRT::intersect_tri<portableRT::Backend::CPU>(tri, miss);

    bool hit_optix = portableRT::intersect_tri<portableRT::Backend::OPTIX>(tri, hit);
    bool miss_optix = portableRT::intersect_tri<portableRT::Backend::OPTIX>(tri, miss);
}
```
