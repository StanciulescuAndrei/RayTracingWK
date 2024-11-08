#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__device__ glm::vec3 randomPointOnSphere(curandState& state){
    float x = 2.0f * curand_uniform(&state) - 1.0f;
    float y = 2.0f * curand_uniform(&state) - 1.0f;
    float z = 2.0f * curand_uniform(&state) - 1.0f;

    return glm::normalize(glm::vec3(x, y, z));
}