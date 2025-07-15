#include <optix.h>
#include <cstdio>

struct Params        
{
    OptixTraversableHandle handle;
    float4*  origins;     
    float4*  directions;
    float*    results;    
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

    unsigned int hit = 0;
    optixTrace( params.handle,
                origin,
                direction,
                0.0f,              
                10000000.0f,           
                0.0f,     
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,           
                hit );
    float t = __uint_as_float(hit);
    params.results[i] = t;

}

extern "C" __global__ void __miss__ms()        {
    optixSetPayload_0(__float_as_uint(INFINITY));
}

extern "C" __global__ void __closesthit__ch() {
    float t = optixGetRayTmax();  
    optixSetPayload_0(__float_as_uint(t));
}