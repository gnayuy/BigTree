#ifndef __GPULZWDECOMPH
#define __GPULZWDECOMPH

#include <stdio.h>
#include <stdlib.h>

#include "cudalzwdecompkernel.cuh"
#include "Image.h"

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

unsigned char *cudaLZWdecompToHost(const char *filestr, int width,int length,int samplesPerPixel,int bytesPerPixel)
{
    unsigned int filesize;
    unsigned char *TiffFile = ReadBinary(filename,&filesize);//+++ Read Tiff File
    TAG tag;
    ReadTiffTag(TiffFile,filesize,&tag); //+++ Set file-information
    //PrintTiffTag(TiffFile,&tag,0);
    if(tag.Compression!=5){
        fprintf(stderr,"Error : NOT LZW comopressed file\n");
        exit(1);
    }


    int OriSize = width*length*samplesPerPixel*bytesPerPixel;


    int StripSize;
    if(samplesPerPixel==1){
        StripSize=tag.ImageWidth*tag.RowsPerStrip;
    }else if(tag.SamplesPerPixel==3){
        StripSize=tag.ImageWidth*tag.RowsPerStrip*3*(tag.PlanarConfiguration==1);
    }
	unsigned char *g_TiffFile; //+++ Input Tiff File
	unsigned char *g_ImageData;//+++ Decompressed Data
	cudaMalloc((void**)&g_TiffFile  ,sizeof(unsigned char)*filesize);
    cudaMalloc((void**)&g_ImageData ,sizeof(unsigned char)*OriSize);
    cudaMemcpy(g_TiffFile,TiffFile,sizeof(unsigned char)*tag.FileSize,cudaMemcpyHostToDevice);

	GPU_TiffLZWDecompression<<<tag.StripNumber,DIMLEN>>>(
											g_TiffFile,
											g_ImageData,
											StripSize,
											tag.StripOffsets_Offset,
											tag.StripByteCounts_Offset,
											tag.ByteOder);
    //+++ GPU LZW Decompression
    if(tag.Predictor==2){
        if(tag.SamplesPerPixel==1){
            int blocknum=(tag.ImageLength+5)/6;
            int laneblocknum=(tag.ImageLength+2)/3;
            GPU_TIFFPredictor_gray<<<blocknum,64>>>(g_ImageData,tag.ImageWidth,tag.ImageLength,laneblocknum);
        }else if(tag.SamplesPerPixel==3){
            if(tag.PlanarConfiguration==1){
                int blocknum=(tag.ImageLength+1)/2;
                int laneblocknum=tag.ImageLength;
                GPU_TIFFPredictor_color<<<blocknum,64>>>(g_ImageData,tag.ImageWidth,tag.ImageLength,laneblocknum);
            }else{
                int blocknum=(tag.ImageLength*3+1)/2;
                int laneblocknum=tag.ImageLength;
                GPU_TIFFPredictor_gray<<<blocknum,64>>>(g_ImageData,tag.ImageWidth,tag.ImageLength,laneblocknum);
            }
        }
    }

    unsigned char *ImageData=(unsigned char *)malloc(sizeof(unsigned char)*OriSize);
    cudaMemcpy(ImageData,g_ImageData,sizeof(unsigned char)*OriSize,cudaMemcpyDeviceToHost);
    //cudaThreadSynchronize();
    cudaFree(g_TiffFile);
    free(TiffFile);
    *Width=tag.ImageWidth;
    *Length=tag.ImageLength;
    *SamplesPerPixel=tag.SamplesPerPixel;
    *PlanarConfiguration=tag.PlanarConfiguration;
    //printf("%s\n",cudaGetErrorString(cudaGetLastError()));
    return ImageData;
}
#endif
