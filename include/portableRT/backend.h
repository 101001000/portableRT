#pragma once

#include <vector>
#include <string>

#include "core.h"
#include "intersect.h"

namespace portableRT {

    using IntersectTriFn = bool (*)(const std::array<float,9>&, const Ray&);

    class Backend{
        Backend() = delete;
    public:
        Backend(BackendType type, std::string name, IntersectTriFn fn){
            this->type = type;
            this->name = name;
            this->fn = fn;
        }
        BackendType type;
        std::string name;
        IntersectTriFn fn;
    };

    inline const Backend* selected_backend = nullptr;
    inline std::vector<Backend> all_backends_;
    inline const std::vector<Backend>& all_backends() { return all_backends_; };

    void select_backend(const Backend& backend){
        selected_backend = &backend;
        intersect_tri_call = backend.fn;
    }

    template<BackendType Type, bool (*Fn)(const std::array<float,9>&, const Ray&)>
    struct RegisterBackend {
        RegisterBackend(const char* name) {
            all_backends_.push_back({Type, name, Fn});
        }
    };

}