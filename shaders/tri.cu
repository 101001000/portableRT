#include <optix.h>
#include <cstdio>

struct Params        
{
    OptixTraversableHandle handle;
    float3  origin;     
    float3  direction;
    int*    result;    
};

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{


    unsigned int hit = 0;
    optixTrace( params.handle,
                params.origin,
                params.direction,
                0.0f,              
                1000.0f,           
                0.0f,     
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,           
                hit );
    *params.result = static_cast<int>( hit );

}

extern "C" __global__ void __miss__ms()        {
    optixSetPayload_0( static_cast<unsigned int>( 0 ) );
}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0( static_cast<unsigned int>( 1 ) );
}