#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "cuda_common/helper_cuda.h"

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>


#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define BLOCK_X 16
#define BLOCK_Y 16


static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void initGLContextAndWindow(GLFWwindow** window){
    
    glfwSetErrorCallback(error_callback);
 
    if (!glfwInit())
        exit(EXIT_FAILURE);
 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
 
    *window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "RayTracingWK", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
 
    glfwSetKeyCallback(*window, key_callback);
    glfwSetFramebufferSizeCallback(*window, framebuffer_size_callback);
 
    glfwMakeContextCurrent(*window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);
}

int main(){
    GLFWwindow* window;
    initGLContextAndWindow(&window);

    /* OpenGL configuration */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 16);      // 4-byte pixel alignment

    glClearColor(0, 0, 0, 0);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    glEnable(GL_BLEND);  
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

    dim3 block(BLOCK_X, BLOCK_Y, 1); // One thread per pixel!
    dim3 grid(SCREEN_WIDTH / BLOCK_X + 1, SCREEN_HEIGHT / BLOCK_Y + 1, 1);

    /* Set up resources for texture writing */
    GLuint pboId;
    GLuint texId;
    GLfloat * imageData = new GLfloat[SCREEN_HEIGHT * SCREEN_WIDTH * 4];

    struct cudaGraphicsResource * cuda_pbo_resource;
    void * d_pbo_buffer = NULL;

    // Initialize the texture
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, (GLvoid*)imageData);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Initialize PBO
    glGenBuffers(1, &pboId);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, SCREEN_HEIGHT * SCREEN_WIDTH * 4 * sizeof(float), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Prepare CUDA interop
    checkCudaErrors(cudaMalloc(&d_pbo_buffer, 4 * SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(float)));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pboId, cudaGraphicsRegisterFlagsNone));
    
    /* Main program loop */
    while (!glfwWindowShouldClose(window))
    {
        /* Clear color and depth buffers */
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        /* Bind the texture and Pixel Buffer */
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
        glBindTexture(GL_TEXTURE_2D, texId);
        
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_FLOAT, 0);

        /* Map the OpenGL resources to a CUDA memory location */
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        float4* dataPointer = nullptr;
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dataPointer, &num_bytes, cuda_pbo_resource));
        assert(num_bytes >= SCREEN_HEIGHT * SCREEN_WIDTH * 4 * sizeof(float));
        assert(dataPointer != nullptr);

        /* Do the rendering here */
        


        /* Unmap the OpenGL resources */
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        /* Draw a quad which covers the entire screen */
        glBindTexture(GL_TEXTURE_2D, texId);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1);
        glTexCoord2f(0.0f, 0.0f);   glVertex3f(-1.0f, -1.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f);   glVertex3f( 1.0f, -1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f);   glVertex3f( 1.0f,  1.0f, 0.0f);
        glTexCoord2f(0.0f, 1.0f);   glVertex3f(-1.0f,  1.0f, 0.0f);
        glEnd();

        /* Unbind the texture and PBO */
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        /* Swap buffers and handle GLFW events */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    /* Unmap resources and free allocated memory */
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteTextures(1, &texId);
    glDeleteBuffers(1, &pboId);
    cudaFree(d_pbo_buffer);

    delete [] imageData;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
