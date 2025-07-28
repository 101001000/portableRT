#include <optix_device.h>
#include <cstdio>
#include <cstdint>

struct alignas(16) HitReg{
    float u;
    float v;
    float t;
    uint32_t primitive_id;
    alignas(4) bool valid;
    float3 p;
};

struct Params        
{
    OptixTraversableHandle handle;
    float4*  origins;     
    float4*  directions;
    HitReg*    results;    
};

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{

    const uint3 idx = optixGetLaunchIndex();
    const int i = idx.x;

    float4 o4 = params.origins[i];
    float4 d4 = params.directions[i];
    float3 origin = make_float3(o4.x, o4.y, o4.z);
    float3 direction = make_float3(d4.x, d4.y, d4.z);

    unsigned int p0, p1, p2, p3;

    optixTrace( params.handle,
                origin,
                direction,
                0.0f,              
                10000000.0f,           
                0.0f,     
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,           
                p0, p1, p2, p3 );
    params.results[i].t = __uint_as_float(p0);
    params.results[i].u = __uint_as_float(p1);
    params.results[i].v = __uint_as_float(p2);
    params.results[i].primitive_id = p3;
    params.results[i].valid = isfinite(__uint_as_float(p0));
    params.results[i].p = make_float3(o4.x + __uint_as_float(p0) * d4.x, o4.y + __uint_as_float(p0) * d4.y, o4.z + __uint_as_float(p0) * d4.z);
}

extern "C" __global__ void __miss__ms()        {
    optixSetPayload_0(__float_as_uint(INFINITY));
    //optixSetPayload_1(0xFFFFFFFFu);
}

extern "C" __global__ void __closesthit__ch() {
    float2 barycentric = optixGetTriangleBarycentrics();
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(__float_as_uint(barycentric.x));
    optixSetPayload_2(__float_as_uint(barycentric.y));
    optixSetPayload_3(optixGetPrimitiveIndex());
}