#pragma once

#include "core.hpp"
#include <numeric>

namespace portableRT{

namespace temp{

class BVH2{
public:
    using IdxTriVector = std::vector<std::pair<uint32_t, Tri>>;
    using AABB = std::pair<std::array<float, 3>, std::array<float, 3>>;


    static AABB empty_aabb(){
        return {{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()}, {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()}};
    }

    struct Node{
        bool is_leaf = false;
        uint32_t tri = -1;
        uint32_t left = -1;
        uint32_t right = -1;
        AABB bounds = empty_aabb();
    };

    static AABB make_aabb(const Tri& tri){
        AABB bounds;
        bounds.first[0] = std::min(tri[0], std::min(tri[3], tri[6]));
        bounds.first[1] = std::min(tri[1], std::min(tri[4], tri[7]));
        bounds.first[2] = std::min(tri[2], std::min(tri[5], tri[8]));
        bounds.second[0] = std::max(tri[0], std::max(tri[3], tri[6]));
        bounds.second[1] = std::max(tri[1], std::max(tri[4], tri[7]));
        bounds.second[2] = std::max(tri[2], std::max(tri[5], tri[8]));
        return bounds;
    }

    static AABB extend_aabb(const AABB& a, const AABB& b){
        AABB bounds;
        bounds.first[0] = std::min(a.first[0], b.first[0]);
        bounds.first[1] = std::min(a.first[1], b.first[1]);
        bounds.first[2] = std::min(a.first[2], b.first[2]);
        bounds.second[0] = std::max(a.second[0], b.second[0]);
        bounds.second[1] = std::max(a.second[1], b.second[1]);
        bounds.second[2] = std::max(a.second[2], b.second[2]);
        return bounds;
    }



    static void divide_sah(const IdxTriVector &input, IdxTriVector &left, IdxTriVector &right) {
        // TODO
        for(int i = 0; i < input.size(); ++i){
            if(i < input.size() / 2){
                left.push_back(input[i]);
            }else{
                right.push_back(input[i]);
            }
        }
    }
    
    static uint32_t build_aux(const IdxTriVector &input, std::vector<Node>& node_vector){

        const uint32_t my_idx = static_cast<uint32_t>(node_vector.size());
        node_vector.emplace_back();      

        if(input.empty()){
            node_vector[my_idx].is_leaf = true;
        }else if(input.size() == 1){
            node_vector[my_idx].is_leaf = true;
            node_vector[my_idx].tri = input[0].first;
            node_vector[my_idx].bounds = make_aabb(input[0].second);
        } else {
            node_vector[my_idx].is_leaf = false;
            AABB bounds = empty_aabb();
            for(auto& tri : input){
                bounds = extend_aabb(bounds, make_aabb(tri.second));
            }
            node_vector[my_idx].bounds = bounds;
            IdxTriVector left;
            IdxTriVector right;
            divide_sah(input, left, right);
            node_vector[my_idx].left = build_aux(left, node_vector);
            node_vector[my_idx].right = build_aux(right, node_vector);
        }
        return my_idx;
    }

    void build(const std::vector<Tri> &tris){

        std::vector<Node> node_vector;

        IdxTriVector idx_tris;

        for(uint32_t i = 0; i < tris.size(); i++){
            m_tris.push_back(tris[i]);
            idx_tris.push_back(std::make_pair(i, tris[i]));
        }

        build_aux(idx_tris, node_vector);

        nodes = new Node[node_vector.size()];
        std::copy(node_vector.begin(), node_vector.end(), nodes); 

    }

    static bool ray_box_intersect(const Ray& ray, const std::array<float, 3>& b_min, const std::array<float, 3>& b_max){
        std::array<float, 3> dirfrac;

        dirfrac[0] = 1.0f / ray.direction[0];
        dirfrac[1] = 1.0f / ray.direction[1];
        dirfrac[2] = 1.0f / ray.direction[2];
      
        float t1 = (b_min[0] - ray.origin[0]) * dirfrac[0];
        float t2 = (b_max[0] - ray.origin[0]) * dirfrac[0];
        float t3 = (b_min[1] - ray.origin[1]) * dirfrac[1];
        float t4 = (b_max[1] - ray.origin[1]) * dirfrac[1];
        float t5 = (b_min[2] - ray.origin[2]) * dirfrac[2];
        float t6 = (b_max[2] - ray.origin[2]) * dirfrac[2];
      
        float tmin =
            std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax =
            std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
      
        if (tmax < 0) {
          return false;
        }

        if (tmin > tmax) {
          return false;
        }
      
        return true;
    }

    std::pair<uint32_t, float> nearest_tri(const Ray& ray){

        constexpr uint32_t k_stacksize = 64;
        uint32_t node_stack[k_stacksize];

        uint32_t stack_idx = 0;
        node_stack[stack_idx++] = 0;

        float t_near = std::numeric_limits<float>::infinity();
        uint32_t nearest_tri_idx = -1;
        
        while(stack_idx > 0){
            if(stack_idx >= k_stacksize){
                std::cout << "OVERFLOW" << std::endl;
                break;
            }
            uint32_t node_idx = node_stack[--stack_idx];
            const Node& node = nodes[node_idx];
            if(!ray_box_intersect(ray, node.bounds.first, node.bounds.second)){
                continue;
            }
            if(node.is_leaf){
                float t = intersect_tri(m_tris[node.tri], ray);
                if(t < t_near){
                    t_near = t;
                    nearest_tri_idx = node.tri;
                }
            } else {
                node_stack[stack_idx++] = node.left;
                node_stack[stack_idx++] = node.right;
            }
        }

        return std::make_pair(nearest_tri_idx, t_near);
    }
private:
    Node* nodes = nullptr;
    std::vector<Tri> m_tris;
};

    
}
}