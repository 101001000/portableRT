#pragma once

#include <array>
#include "core.h"

namespace portableRT {

    
    using IntersectTriFn = bool (*)(const std::array<float,9>&, const Ray&);
    
    
    inline IntersectTriFn intersect_tri_call = nullptr;
    
    
    inline bool intersect_tri(const std::array<float,9>& v, const Ray& r){
        return intersect_tri_call(v, r); 
    }
    
    template<BackendType B>
    bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);

}
