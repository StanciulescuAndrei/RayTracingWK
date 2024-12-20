#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include "ray.cuh"

class Camera{
    public:
        Camera(glm::vec3 position, float focalLength, int imageWidth, int imageHeight){
            pos = position;
            focal = focalLength;

            viewport_u = glm::vec3((2.0f * imageWidth) / imageHeight, 0.0f, 0.0f);
            viewport_v = glm::vec3(0.0f, -2.0f, 0.0f);

            pixel_delta_u = viewport_u / (float)imageWidth;
            pixel_delta_v = viewport_v / (float)imageHeight;

            viewport_upper_left = pos - glm::vec3(0.0f, 0.0f, focal) - viewport_u/2.0f - viewport_v/2.0f;
            pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
        }

        __device__ Ray getPixelRay(int x, int y){
            glm::vec3 pixel_center = pixel00_loc + (static_cast<float>(x) * pixel_delta_u) + (static_cast<float>(y) * pixel_delta_v);
            glm::vec3 ray_direction = pixel_center - pos;

            return Ray(pos, ray_direction);
        }

        __device__ void getPixelMultisample(int x, int y, Ray * rayBuffer, const int samples){
            glm::vec3 pixel_center;
            glm::vec3 ray_direction;
            int dSamples = sqrtf(static_cast<float>(samples));
            glm::vec3 multi_delta_u = pixel_delta_u / static_cast<float>(dSamples);
            glm::vec3 multi_delta_v = pixel_delta_u / static_cast<float>(dSamples);

            for(int di = 0; di < dSamples; di++){
                for(int dj = 0; dj < dSamples; dj++){
                    pixel_center = pixel00_loc + (static_cast<float>(x) * pixel_delta_u + static_cast<float>(di)* multi_delta_u) + (static_cast<float>(y) * pixel_delta_v + static_cast<float>(dj) * multi_delta_v);
                    ray_direction = pixel_center - pos;
                    rayBuffer[di + dj * dSamples] = Ray(pos, ray_direction);
                }
            }
        }

    private:
        glm::vec3 pos;
        float focal;
        glm::vec3 viewport_u;
        glm::vec3 viewport_v;

        glm::vec3 pixel_delta_u;
        glm::vec3 pixel_delta_v;

        glm::vec3 viewport_upper_left;
        glm::vec3 pixel00_loc;
};

#endif