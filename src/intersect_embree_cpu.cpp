#include <embree4/rtcore.h>
#include <limits>
#include <math.h>
#include <stdio.h>

#include "../include/portableRT/intersect_embree_cpu.hpp"

void errorFunctionCpu(void *userPtr, enum RTCError error, const char *str) {
  printf("error %d: %s\n", error, str);
}

RTCDevice initializeDevice() {
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(device, errorFunctionCpu, NULL);
  return device;
}

bool castRay(RTCScene scene, float ox, float oy, float oz, float dx, float dy,
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
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(scene, &rayhit);

  return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

namespace portableRT {

void EmbreeCPUBackend::initializeScene() {

  rtcSetSceneFlags(m_scene, RTC_SCENE_FLAG_DYNAMIC);
  m_tri = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  float *vertices =
      (float *)rtcSetNewGeometryBuffer(m_tri, RTC_BUFFER_TYPE_VERTEX, 0,
                                       RTC_FORMAT_FLOAT3, 3 * sizeof(float), 3);

  unsigned *indices = (unsigned *)rtcSetNewGeometryBuffer(
      m_tri, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned),
      1);

  if (vertices && indices) {
    vertices[0] = 0;
    vertices[1] = 0;
    vertices[2] = 0;
    vertices[3] = 0;
    vertices[4] = 0;
    vertices[5] = 0;
    vertices[6] = 0;
    vertices[7] = 0;
    vertices[8] = 0;

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
  }
  rtcCommitGeometry(m_tri);
  rtcAttachGeometry(m_scene, m_tri);
  rtcCommitScene(m_scene);
}

bool EmbreeCPUBackend::intersect_tri(const std::array<float, 9> &tri,
                                     const Ray &ray) {

  float *vertices =
      (float *)rtcGetGeometryBufferData(m_tri, RTC_BUFFER_TYPE_VERTEX, 0);
  for (int i = 0; i < 9; i++) {
    vertices[i] = tri[i];
  }
  rtcCommitGeometry(m_tri);
  rtcCommitScene(m_scene);

  bool res = castRay(m_scene, ray.origin[0], ray.origin[1], ray.origin[2],
                     ray.direction[0], ray.direction[1], ray.direction[2]);

  return res;
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
  initializeScene();
}

void EmbreeCPUBackend::shutdown() {
  rtcReleaseGeometry(m_tri);
  rtcReleaseScene(m_scene);
  rtcReleaseDevice(m_device);
}

} // namespace portableRT