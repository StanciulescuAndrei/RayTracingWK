#ifndef HITTABLE_H
#define HITTABLE_H

#include <glm/glm.hpp>
#include "ray.cuh"

class HitRecord{
    public:
        glm::vec3 p;
        glm::vec3 normal;
        float t;
};

class HittableEntity{
    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hitRecord) const = 0; 
};

class Sphere : HittableEntity {
    public:
        __device__ Sphere(const glm::vec3& _position, float _radius) : position(_position), radius(_radius) {}

        __device__ bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hitRecord) const override {
            glm::vec3 oc = position - ray.origin();
            float a = glm::dot(ray.direction(), ray.direction());
            float b = -2.0f * glm::dot(oc, ray.direction());
            float c = glm::dot(oc, oc) - radius * radius;

            float discriminant = b * b - 4.0f * a * c;
            if(discriminant < 0){
                return false;
            }
            float discriminant_sq = glm::sqrt(discriminant);

            float root = (-b - discriminant) / (2.0f * a);
            if(root < t_min || root > t_max){
                root = (-b + discriminant) / (2.0f * a);
                if(root < t_min || root > t_max){
                    return false;
                }
            }

            hitRecord.t = root;
            hitRecord.p = ray.at(root);
            hitRecord.normal = (hitRecord.p - position) / radius;
            return true;
        }

    private:
        glm::vec3 position;
        float radius;
};

#endif