

#include "camera.h"
#include "hittable.h"

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

__global__ void render(int2 resolution, float4 * data, Camera camera, HitableList ** hList){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    int x = idx % resolution.x; // Horizontal positioning
    int y = idx / resolution.x; // Vertical positioning

    if(y >= resolution.y) return;

    HitRecord hitRecord;
    glm::vec3 out_color;

    if(hList[0]->hit(camera.getPixelRay(x, y), 0.0f, 1000000.0f, hitRecord)){
        out_color = hitRecord.normal * 0.5f + 0.5f;
    }
    else{
        glm::vec3 unit_direction = glm::normalize(camera.getPixelRay(x, y).direction());
        float a = 0.5*(unit_direction[1] + 1.0);
        out_color =  (1.0f-a)*glm::vec3(1.0f, 1.0f, 1.0f) + a*glm::vec3(0.5f, 0.7f, 1.0f);
    }

    data[(resolution.y - y - 1) * resolution.x + x] = {out_color[0], out_color[1], out_color[2], 1.0f};

}