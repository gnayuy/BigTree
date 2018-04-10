// BigTree
// Reformatting large-scale dataset into a hierarchical file organization
// gnayuy, 3/30/2018
//
//
// Here is simple implementation to demonstrate the idea of "BigTree: high-performance hierarchical tree construction for large image data sets"
//


//
# include <stdio.h>
# include <stdlib.h>
#include <iostream>

#include "BigTree.h"

//
int main (int argc, const char *argv[])
{
    // Usage:
    // BigTree -i <input_DIR> -o <output_DIR> -n <Number_of_Resolutions_of_TMITREE>

    // assuming input 2D TIFF (LZW compressed) images and convert to 3D TIFF blocks
    // 3D block with 256x256x256

    //
    bool usingGPU = true;

    // BigTree
    BigTree bt(argv[1], argv[2], atoi(argv[3]), usingGPU);

    //
    return 0;
}
