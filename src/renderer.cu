#include <curand_kernel.h>
#include "camera.h"
#include "hittable.h"
#include <glm/gtc/random.hpp>
#include "utils.cuh"

const int numSceneElements = 4;

__global__ void populateScene(HitableList** hList, Sphere ** hittableBuffer, const int numElements){
    if(threadIdx.x != 0 || blockIdx.x != 0){
        return;
    }
    
    hittableBuffer[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
    hittableBuffer[1] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f);
    hittableBuffer[2] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f);
    hittableBuffer[3] = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f);

    *hList = new HitableList(hittableBuffer, numElements);
}

const int maxDepth = 10;

__device__ glm::vec3 rayColor(const Ray& ray, HitableList ** hList, int depth, curandState& state){
    if(depth == maxDepth){
        return glm::vec3(0.0f);
    }

    HitRecord hitRecord;

    if(hList[0]->hit(ray, 0.001f, 1000000.0f, hitRecord)){
        glm::vec3 nextDirection = hitRecord.normal + randomPointOnSphere(state);
        nextDirection = glm::normalize(nextDirection);
        return 0.5f * rayColor(Ray(hitRecord.p, nextDirection), hList, depth + 1, state);
    }
    else{
        glm::vec3 unit_direction = glm::normalize(ray.direction());
        float a = 0.5*(unit_direction[1] + 1.0);
        return (1.0f-a)*glm::vec3(1.0f, 1.0f, 1.0f) + a*glm::vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void render(int2 resolution, float4 * data, Camera camera, HitableList ** hList){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    int x = idx % resolution.x; // Horizontal positioning
    int y = idx / resolution.x; // Vertical positioning

    if(y >= resolution.y) return;

    curandState localState;
    curand_init(12345, idx, 0, &localState);
    
    glm::vec3 out_color(0.0f);
    const int nMultiSamples = 16;
    Ray multiSampleRays[nMultiSamples];
    camera.getPixelMultisamplex4(x, y, multiSampleRays, nMultiSamples);
    for(int i = 0; i < nMultiSamples; i++){
        out_color += 1.0f / nMultiSamples * rayColor(multiSampleRays[i], hList, 0, localState);
    }    

    data[(resolution.y - y - 1) * resolution.x + x] = {out_color[0], out_color[1], out_color[2], 1.0f};

}