#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "utils.cuh"
#include <curand_kernel.h>

class Material{
    public:
        __device__  virtual bool scatter(const Ray& rayIn, HitRecord& hitRecord, glm::vec3& attenuation, Ray& scatteredRay, curandState& state) const {
            return false;
        }
};

class Lambertian : public Material{
    public: 
        __device__ __host__ Lambertian(const glm::vec3& albedo) : albedo(albedo) {}
        __device__  bool scatter(const Ray& rayIn, HitRecord& hitRecord, glm::vec3& attenuation, Ray& scatteredRay, curandState& state) const override{
            glm::vec3 nextDirection = hitRecord.normal + randomPointOnSphere(state);
            nextDirection = glm::normalize(nextDirection);
            scatteredRay = Ray(hitRecord.p, nextDirection);
            attenuation = albedo;
            return true;
        }

    private:
        glm::vec3 albedo;
};

class Metallic : public Material{
    public: 
        __device__ __host__ Metallic(const glm::vec3& albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}
        __device__  bool scatter(const Ray& rayIn, HitRecord& hitRecord, glm::vec3& attenuation, Ray& scatteredRay, curandState& state) const override{
            glm::vec3 reflected = reflect(rayIn.direction(), hitRecord.normal) + randomPointOnSphere(state) * fuzz;
            if(glm::dot(hitRecord.normal, reflected) < 0.0f){
                return false;
            }
            scatteredRay = Ray(hitRecord.p, reflected);
            attenuation = albedo;
            return true;
        }

    private:
        glm::vec3 albedo;
        float fuzz;
};

#endif