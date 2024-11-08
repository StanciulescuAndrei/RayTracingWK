#ifndef RAY_H
#define RAY_H
#include <iostream>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>

class Ray{
    public:
        __device__ __host__ Ray() {}
        __device__ __host__ Ray(glm::vec3 origin, glm::vec3 direction) : orig(origin), dir(direction) {} 

        __device__ __host__ const glm::vec3& direction() const {return dir;}

        __device__ __host__ const glm::vec3& origin() const {return orig;}

        __device__ __host__ glm::vec3 at(float t) const {return orig + t * dir;}


    private:
        glm::vec3 orig;
        glm::vec3 dir;
};

#endif