#ifndef __GPULZWDECOMPH
#define __GPULZWDECOMPH

#include <stdio.h>
#include <stdlib.h>

#include "cudalzwdecompkernel.cuh"
#include "culzw.h"

void gpu_init()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(nDevices<1){
        fprintf(stderr,"Warning:There is no GPU.\n");
    }
    unsigned char *g_tmp;
    cudaMalloc((void**)&g_tmp ,1);
    cudaFree(g_tmp);
}

int cudaLZWdecompToHost(unsigned char *compressedImage, int compressedSize, unsigned char *&p, unsigned int width, unsigned int length,
                        unsigned int samplesPerPixel, unsigned int bytesPerPixel, unsigned int rowsPerStrip, unsigned int stripOffsets, unsigned int stripByteCounts)
{
    //
    int imageSize = width*length*samplesPerPixel*bytesPerPixel;

    //
    int StripSize;
    if(samplesPerPixel==1)
    {
        StripSize=width*rowsPerStrip;
    }
    else
    {
        // to do
    }

    unsigned int stripNumber = *((unsigned int *)(compressedImage+4));

    // host -> device
    unsigned char *g_TiffFile; // Input Tiff File
    unsigned char *g_ImageData;// Decompressed Data
    cudaMalloc((void**)&g_TiffFile  ,sizeof(unsigned char)*compressedSize);
    cudaMalloc((void**)&g_ImageData ,sizeof(unsigned char)*imageSize);
    cudaMemcpy(g_TiffFile,compressedImage,sizeof(unsigned char)*compressedSize,cudaMemcpyHostToDevice);

    GPU_TiffLZWDecompression<<<stripNumber,DIMLEN>>>(g_TiffFile, g_ImageData, StripSize, stripOffsets, stripByteCounts, compressedImage[0]);

    // GPU LZW Decompression
    if(samplesPerPixel==1)
    {
        int blocknum=(length+5)/6;
        int laneblocknum=(width+2)/3;
        GPU_TIFFPredictor_gray<<<blocknum,64>>>(g_ImageData, width, length, laneblocknum);
    }
    else
    {
        // to do
    }

    // device -> host
    cudaMemcpy(p,g_ImageData,sizeof(unsigned char)*imageSize,cudaMemcpyDeviceToHost);

    //cudaThreadSynchronize();
    cudaFree(g_TiffFile);
    free(compressedImage);

    //
    return 0;
}
#endif
