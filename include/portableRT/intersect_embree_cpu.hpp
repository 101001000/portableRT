#pragma once
#include <embree4/rtcore.h>
#include <limits>
#include <math.h>
#include <stdio.h>

#include "core.hpp"

void errorFunctionCpu(void *userPtr, enum RTCError error, const char *str) {
  printf("error %d: %s\n", error, str);
}

/*
 * Embree has a notion of devices, which are entities that can run
 * raytracing kernels.
 * We initialize our device here, and then register the error handler so that
 * we don't miss any errors.
 *
 * rtcNewDevice() takes a configuration string as an argument. See the API docs
 * for more information.
 *
 * Note that RTCDevice is reference-counted.
 */
RTCDevice initializeDevice() {
  RTCDevice device = rtcNewDevice(NULL);

  if (!device)
    printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

  rtcSetDeviceErrorFunction(device, errorFunctionCpu, NULL);
  return device;
}

/*
 * Cast a single ray with origin (ox, oy, oz) and direction
 * (dx, dy, dz).
 */
bool castRay(RTCScene scene, float ox, float oy, float oz, float dx, float dy,
             float dz) {
  /*
   * The ray hit structure holds both the ray and the hit.
   * The user must initialize it properly -- see API documentation
   * for rtcIntersect1() for details.
   */
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

  /*
   * There are multiple variants of rtcIntersect. This one
   * intersects a single ray with the scene.
   */
  rtcIntersect1(scene, &rayhit);

  return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

namespace portableRT {

class EmbreeCPUBackend : public InvokableBackend<EmbreeCPUBackend> {
public:
  EmbreeCPUBackend() : InvokableBackend(BackendType::EMBREE_CPU, "Embree CPU") {
    static RegisterBackend reg(*this);
  }

  void initializeScene() {

    rtcSetSceneFlags(m_scene, RTC_SCENE_FLAG_DYNAMIC);
    m_tri = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float *vertices = (float *)rtcSetNewGeometryBuffer(
        m_tri, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
        3);

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

    /*
     * You must commit geometry objects when you are done setting them up,
     * or you will not get any intersections.
     */
    rtcCommitGeometry(m_tri);

    /*
     * In rtcAttachGeometry(...), the scene takes ownership of the geom
     * by increasing its reference count. This means that we don't have
     * to hold on to the geom handle, and may release it. The geom object
     * will be released automatically when the scene is destroyed.
     *
     * rtcAttachGeometry() returns a geometry ID. We could use this to
     * identify intersected objects later on.
     */
    rtcAttachGeometry(m_scene, m_tri);

    /*
     * Like geometry objects, scenes must be committed. This lets
     * Embree know that it may start building an acceleration structure.
     */
    rtcCommitScene(m_scene);
  }

  bool intersect_tri(const std::array<float, 9> &tri, const Ray &ray) {

    float *vertices =
        (float *)rtcGetGeometryBufferData(m_tri, RTC_BUFFER_TYPE_VERTEX, 0);
    for (int i = 0; i < 9; i++) {
      vertices[i] = tri[i];
    }
    rtcCommitGeometry(m_tri);
    rtcCommitScene(m_scene);

    /* This will hit the triangle at t=1. */
    bool res = castRay(m_scene, ray.origin[0], ray.origin[1], ray.origin[2],
                       ray.direction[0], ray.direction[1], ray.direction[2]);

    /* Though not strictly necessary in this example, you should
     * always make sure to release resources allocated through Embree. */

    return res;
  }

  bool is_available() const override {
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

  void init() override {
    m_device = initializeDevice();
    m_scene = rtcNewScene(m_device);
    initializeScene();
  }

  void shutdown() override {
    rtcReleaseGeometry(m_tri);
    rtcReleaseScene(m_scene);
    rtcReleaseDevice(m_device);
  }

private:
  RTCDevice m_device;
  RTCScene m_scene;
  RTCGeometry m_tri;
};

static EmbreeCPUBackend embreecpu_backend;

} // namespace portableRT