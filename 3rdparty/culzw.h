#ifndef __CULZW_H
#define __CULZW_H

//
void gpu_init();
unsigned char *cudaLZWdecompToHost(const char *filestr, int width,int length,int samplesPerPixel,int bytesPerPixel);

#endif // __CULZW_H
