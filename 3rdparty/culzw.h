#ifndef __CULZW_H
#define __CULZW_H

//
void gpu_init();
int cudaLZWdecompToHost(unsigned char *compressedImage, int compressedSize, unsigned char *&p, unsigned int width, unsigned int length,
                        unsigned int samplesPerPixel, unsigned int bytesPerPixel, unsigned int rowsPerStrip, unsigned int stripOffsets, unsigned int stripByteCounts);

#endif // __CULZW_H
