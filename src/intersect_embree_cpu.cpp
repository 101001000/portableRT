#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <thread>

#include "../include/portableRT/intersect_embree_cpu.hpp"

void errorFunctionCpu(void *userPtr, enum RTCError error, const char *str) {
  printf("error %d: %s\n", error, str);
}

RTCDevice initializeDevice() {
  RTCDevice device = rtcNewDevice(nullptr);

  if (!device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(nullptr));

  rtcSetDeviceErrorFunction(device, errorFunctionCpu, nullptr);
  return device;
}

std::tuple<float, uint32_t> castRay(RTCScene scene, float ox, float oy, float oz, float dx, float dy,
              float dz) {

  struct RTCRayHit rayhit;
  rayhit.ray.org_x = ox;
  rayhit.ray.org_y = oy;
  rayhit.ray.org_z = oz;
  rayhit.ray.dir_x = dx;
  rayhit.ray.dir_y = dy;
  rayhit.ray.dir_z = dz;
  rayhit.ray.tnear = 0;
  rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rayhit.ray.mask = -1;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(scene, &rayhit);

  return std::make_tuple(rayhit.ray.tfar, rayhit.hit.primID);
}

namespace portableRT {

void EmbreeCPUBackend::set_tris(const Tris &tris) {

  if (m_geom_id != static_cast<uint>(-1)) {
    rtcDetachGeometry(m_scene, m_geom_id);
    rtcReleaseGeometry(m_tri);
  }

  m_tri = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  float *vertices = (float *)rtcSetNewGeometryBuffer(
      m_tri, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(float) * 3,
      tris.size() * 3);

  unsigned *indices = (unsigned *)rtcSetNewGeometryBuffer(
      m_tri, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3,
      tris.size());

  for (int i = 0; i < tris.size(); ++i) {
    for (int j = 0; j < 3; ++j)
      indices[i * 3 + j] = i * 3 + j;
    for (int j = 0; j < 9; ++j)
      vertices[i * 9 + j] = tris[i][j];
  }

  rtcCommitGeometry(m_tri);
  m_geom_id = rtcAttachGeometry(m_scene, m_tri);
  rtcCommitScene(m_scene);
}

void check_hit(int i, RTCScene scene, const std::vector<Ray> &rays, std::vector<HitReg> &hits, int N){
  for(int r = 0; r < N; r++){
    auto [t, geomID] = castRay(scene, rays[i + r].origin[0], rays[i + r].origin[1], rays[i + r].origin[2],
                      rays[i + r].direction[0], rays[i + r].direction[1], rays[i + r].direction[2]);
    hits[i + r].t = t;
    hits[i + r].primitive_id = geomID;
  }
}

std::vector<HitReg>
EmbreeCPUBackend::nearest_hits(const std::vector<Ray> &rays) {

  std::vector<HitReg> hits;
  hits.resize(rays.size());

  clear_affinity();

  unsigned n = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  threads.reserve(n);


  int rays_per_thread = rays.size() / n;

  for (unsigned i = 0; i < n; ++i){
    threads.emplace_back(check_hit,
                         i * rays_per_thread,
                         m_scene,
                         std::cref(rays), 
                         std::ref(hits), rays_per_thread);  
    }

  for (auto &thread : threads) thread.join();

  check_hit(n * rays_per_thread, m_scene, rays, hits, rays.size() % n);

  return hits;
}

bool EmbreeCPUBackend::is_available() const {
  RTCDevice dev = rtcNewDevice(nullptr);
  if (!dev)
    return false;

  RTCError err = rtcGetDeviceError(dev);
  if (err != RTC_ERROR_NONE) {
    rtcReleaseDevice(dev);
    return false;
  }
  rtcReleaseDevice(dev);
  return true;
}

void EmbreeCPUBackend::init() {
  m_device = initializeDevice();
  m_scene = rtcNewScene(m_device);
}

void EmbreeCPUBackend::shutdown() {
  rtcReleaseScene(m_scene);
  rtcReleaseDevice(m_device);
}

std::string EmbreeCPUBackend::device_name() const {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto pos = line.find(':');
      if (pos != std::string::npos) {
        size_t start = line.find_first_not_of(" \t", pos + 1);
        return line.substr(start);
      }
    }
  }
  return "unsupported";
}

} // namespace portableRT