/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 //This code was reuse by Mohd Hakimie to be use with Smalltalk

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// MPI include
#include <mpi.h>
using namespace std;

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SobelFilter_kernels.h"

// includes, project
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

const char *filterMode[] =
{
    "No Filtering",
    "Sobel Texture",
    NULL
};

//for MPI uses
int flag, signal;
MPI_Request request;
MPI_Comm parentcomm;

void cleanup(void);
void initializeData(char *file) ;

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY     10 //ms

const char *sSDKsample = "CUDA Sobel Edge-Detection";

static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

// Code to handle Auto verification
const int frameCheckNumber = 4;
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = NULL;
unsigned int g_Bpp;
unsigned int g_Index = 0;

bool g_bQAReadback = false;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
unsigned char *pixels = NULL;  // Image pixel data on the host
float imageScale = 1.f;        // Image exposure
enum SobelDisplayMode g_SobelDisplayMode;

int *pArgc   = NULL;
char **pArgv = NULL;

//extern "C" void runAutoTest(int argc, char **argv);

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)


// This is the normal display path
void display(void)
{
    sdkStartTimer(&timer);
    char temp[256];

    // Sobel operation
    Pixel *data = NULL;

    // map PBO to get CUDA device pointer
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes, cuda_pbo_resource);

    sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight,
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();

    sdkStopTimer(&timer);

    if (flag == 1 && signal == 1) {
        glutDestroyWindow(glutGetWindow());
        glutCloseFunc(cleanup);
        MPI_Comm_disconnect(&parentcomm);
        MPI_Finalize();
        return;
        
    }
    else if (flag == 1 && signal == 2) { 
        double startTime, endTime;
        startTime = MPI_Wtime();
        g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
        sprintf(temp, "CUDA Edge Detection (%s)", filterMode[g_SobelDisplayMode]);
        glutSetWindowTitle(temp);
        endTime = MPI_Wtime();
        double totalTime = endTime - startTime;
        MPI_Send(&totalTime, 1, MPI_DOUBLE, 0, 0, parentcomm);
        MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
        flag = 0;
    }
    else if (flag == 1 && signal == 3) {
        g_SobelDisplayMode = SOBELDISPLAY_IMAGE;
        sprintf(temp, "CUDA Edge Detection (%s)", filterMode[g_SobelDisplayMode]);
        glutSetWindowTitle(temp);
        MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
        flag = 0;
    }
    else if (flag == 1 && signal == 4) {
        imageScale += 0.1f;
        printf("brightness = %4.2f\n", imageScale);
        MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
        flag = 0;
    }
    else if (flag == 1 && signal == 5) {
        imageScale -= 0.1f;
        printf("brightness = %4.2f\n", imageScale);
        MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);
        flag = 0;
    }
    else {
        
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        
    }
}

void timerEvent(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


void reshape(int x, int y)
{
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteTexture();

    sdkDeleteTimer(&timer);

}

void initializeData(char *file)
{
    GLint bsize;
    unsigned int w, h;
    size_t file_length= strlen(file);
    sdkLoadPGM<unsigned char>(file, &pixels, &w, &h);
    g_Bpp = 1;
    imWidth = (int)w;
    imHeight = (int)h;
    setupTexture(imWidth, imHeight, pixels, g_Bpp);

    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);

    if (!g_bQAReadback)
    {
        // use OpenGL Path
        glGenBuffers(1, &pbo_buffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     g_Bpp * sizeof(Pixel) * imWidth * imHeight,
                     pixels, GL_STREAM_DRAW);

        glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

        if ((GLuint)bsize != (g_Bpp * sizeof(Pixel) * imWidth * imHeight))
        {
            printf("Buffer object (%d) has incorrect size (%d).\n", (unsigned)pbo_buffer, (unsigned)bsize);

            exit(EXIT_FAILURE);
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // register this buffer object with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));

        glGenTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp==1) ? GL_LUMINANCE : GL_BGRA),
                     imWidth, imHeight,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
    }
}




void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("CUDA Edge Detection");

    if (!isGLVersionSupported(1,5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    char buf[256];
    int my_rank, num_procs;
   

    /* Initialize the infrastructure necessary for communication */
    MPI_Init(&argc, &argv);

    /* Identify this process */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many total processes are active */
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
    MPI_Comm_get_parent(&parentcomm);
    //MPI_Recv(buf, sizeof(buf), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Bcast(&buf, sizeof(buf), MPI_CHAR, 0, parentcomm);
    // MPI_Comm_disconnect(&parentcomm);


    MPI_Ibcast(&signal, 1, MPI_INT, 0, parentcomm, &request);


#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    

    initGL(&argc, argv);
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    char* temp = buf;
    initializeData(temp);
    
    fflush(stdout);

    glutCloseFunc(cleanup);

    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMainLoop();
    
    
}
