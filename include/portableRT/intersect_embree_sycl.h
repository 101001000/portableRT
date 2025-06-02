#pragma once
#include <memory>
#include <embree4/rtcore.h>

#include "core.h"
#include "backend.h"

namespace portableRT{

struct EmbreeSYCLBackendImpl;

class EmbreeSYCLBackend : public InvokableBackend<EmbreeSYCLBackend> {
public:
    EmbreeSYCLBackend();
    ~EmbreeSYCLBackend();

    bool intersect_tri(const std::array<float,9>& vertices, const Ray& ray);
    bool is_available() const override;
    void init() override;
    void shutdown() override;
private:
    void initializeScene();

    // Ugly solution to overcome the issues with forward declarations of the sycl type alias in the intel implementation
    // I need to move the sycl import to the .cpp instead the .h so the hip compiler ignores the sycl headers. Other option is 
    // to pack the sycl headers, but that would break sycl dependency.
    std::unique_ptr<EmbreeSYCLBackendImpl> m_impl; 

    RTCDevice m_rtcdevice;
    RTCScene m_rtcscene;
    RTCTraversable m_rtctraversable;
    RTCGeometry m_tri;
};


static EmbreeSYCLBackend embreesycl_backend;

}