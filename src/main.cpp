// BigTree
// gnayuy, 4/1/2019
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
    // BigTree <input_DIR> <output_DIR> <Number_of_Resolutions_of_TMITREE> <Traced_Neuron(SWC)>

    //
    if(argc<2)
    {
        cout<<"bigtree version 1.0\n";
        cout<<"bigtree -h\n";
        return 0;
    }

    //
    if(argc==3)
    {
        BigTree bigtree(argv[1], argv[2]); // 3 layers
    }
    else if(argc==4)
    {
        BigTree bigtree(argv[1], argv[2], atoi(argv[3]));
    }
    else
    {
        BigTree bigtree(argv[1], argv[2], atoi(argv[3]), argv[4]);
    }

    //
    return 0;
}
