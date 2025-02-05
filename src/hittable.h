#ifndef HITTABLE_H
#define HITTABLE_H

#include <glm/glm.hpp>
#include "ray.cuh"

class Material;

class HitRecord{
    public:
        Material * material;
        glm::vec3 p;
        glm::vec3 normal;
        float t;
};

class HitableEntity{
    public:
    __device__ virtual bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hitRecord) const = 0; 
};

class Sphere : public HitableEntity {
    public:
        __device__ Sphere(const glm::vec3& _position, float _radius, Material * _material) : position(_position), radius(_radius), material(_material) {}

        __device__ bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hitRecord) const override {
            glm::vec3 oc = position - ray.origin();
            float a = glm::dot(ray.direction(), ray.direction());
            float b = -2.0f * glm::dot(oc, ray.direction());
            float c = glm::dot(oc, oc) - radius * radius;

            float discriminant = b * b - 4.0f * a * c;
            if(discriminant < 0.0f){
                return false;
            }
            float discriminant_sq = glm::sqrt(discriminant);

            float root = (-b - discriminant_sq) / (2.0f * a);
            if(root < t_min || root > t_max){
                root = (-b + discriminant_sq) / (2.0f * a);
                if(root < t_min || root > t_max){
                    return false;
                }
            }

            hitRecord.t = root;
            hitRecord.p = ray.at(root);
            hitRecord.normal = glm::normalize((hitRecord.p - position));
            if(glm::dot(hitRecord.normal, ray.direction()) > 0.0f){
                hitRecord.normal = -hitRecord.normal;
            }
            hitRecord.material = material;
            return true;
        }

        __device__ void updatePosition(glm::vec3 & newPosition){
            position = newPosition;
        }

        __device__ glm::vec3 getPosition(){
            return position;
        }

        __device__ void deleteMaterial(){
            delete material;
        }

    private:
        glm::vec3 position;
        float radius;
        Material * material;
};

class HitableList : public HitableEntity {
    public:     

    __device__ __host__ HitableList(Sphere ** dataBuffer, const size_t size){
        hittableEntitiesList = dataBuffer;
        numEntities = size;
    }

    __device__ void setEntity(int idx, Sphere* hEntity){
        if(idx >= numEntities){
            return;
        }
        
        hittableEntitiesList[idx] = hEntity;
    }

    __device__ size_t getNumberOfEntities(){
        return numEntities;
    }

    __device__ Sphere * getEntity(int idx){
        if(idx < numEntities){
            return hittableEntitiesList[idx];
        }
        else{
            return nullptr;
        }
    }

    __device__ bool hit(const Ray& ray, float t_min, float t_max, HitRecord& hitRecord) const override {
        int closestHit = t_max;
        bool hitAny = false;
        HitRecord _hitRecord;
        for(int i = 0; i < numEntities; i++){
            if(hittableEntitiesList[i]->hit(ray, t_min, closestHit, _hitRecord)){
                hitAny = true;
                closestHit = _hitRecord.t;
                hitRecord = _hitRecord;
            }
        }
        return hitAny;
    }

    private:
        size_t numEntities;
        Sphere ** hittableEntitiesList;
    
};

#endif