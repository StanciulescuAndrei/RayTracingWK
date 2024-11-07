

#include "camera.h"

__global__ void render(int2 resolution, float4 * data, Camera camera){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    int x = idx % resolution.x; // Horizontal positioning
    int y = idx / resolution.x; // Vertical positioning

    if(y >= resolution.y) return;

    glm::vec3 unit_direction = glm::normalize(camera.getPixelRay(x, y).direction());
    float a = 0.5*(unit_direction[1] + 1.0);
    glm::vec3 out_color =  (1.0f-a)*glm::vec3(1.0f, 1.0f, 1.0f) + a*glm::vec3(0.5f, 0.7f, 1.0f);

    data[(resolution.y - y - 1) * resolution.x + x] = {out_color[0], out_color[1], out_color[2], 1.0f};

}