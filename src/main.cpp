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
    /// Interface:
    // BigTree(string inputdir, string outputdir, int scales=3, string neuron="", int numImages=16, unsigned int bsx=256, unsigned int bsy=256, unsigned int bsz=256);
    /// Usage:
    // BigTree <input_DIR> <output_DIR> <Number_of_Resolutions_of_TMITREE> <Traced_Neuron(SWC)>

    //
    if(argc<2)
    {
        cout<<"BigTree: Big Geometric Tree, version 1.0\n";
        cout<<"BigTree -h\n";
        cout<<"Default parameters:"<<endl;
        cout<<"  3D Block Size: 256X256X256"<<endl;
        cout<<"  Load 16 2D-Images at a time"<<endl;
        cout<<"  Construct BigTree with 3 scales"<<endl;

        cout<<"------"<<endl;
        cout<<"Examples:"<<endl;
        cout<<"1. Construct BigTree of whole dataset into 3 scales: BigTree inputDir outputDir"<<endl;
        cout<<"2. Construct BigTree of whole dataset into n scales: BigTree inputDir outputDir n"<<endl;
        cout<<"3. Construct BigTree of whole dataset into 3 scales with 3D block size (x,y,z): BigTree inputDir outputDir x y z"<<endl;
        cout<<"3. Construct BigTree of whole dataset into n scales with 3D block size (x,y,z): BigTree inputDir outputDir n x y z"<<endl;
        cout<<"4. Construct BigTree for neuron reconstruction: BigTree inputDir outputDir neuron.swc"<<endl;

        //
        return 0;
    }

    //
    int numImages=32;

    cout<<"max load "<<numImages<<" 2D images each time"<<endl;

    //
    if(argc==3)
    {
        BigTree bigtree(argv[1], argv[2]); // 3 layers
        return 0;
    }
    else if(argc==4)
    {
        string str = argv[3];

        size_t found = str.find_last_of('.');
        if(found!=std::string::npos)
        {
            string suffix = str.substr(str.find_last_of('.')+1);
            int n = suffix.size() - 3;

            if(n>=0)
            {
                if(suffix.substr(n) == "swc")
                {
                    BigTree bigtree(argv[1], argv[2], 3, str); // big geometric tree
                    return 0;
                }
            }
        }
        else
        {
            int n = atoi(argv[3]);

            if(n>0 && n<10)
            {
                BigTree bigtree(argv[1], argv[2], n, "", 16, 256, 256, 256, 0, 2); // bigtree conversion
                return 0;
            }
        }
    }
    else if(argc==5)
    {
        BigTree bigtree(argv[1], argv[2], atoi(argv[3]), argv[4]);
        return 0;
    }
    else if(argc==6)
    {
        BigTree bigtree(argv[1], argv[2], 3, "", numImages, atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
        return 0;
    }
    else if(argc==7)
    {
        BigTree bigtree(argv[1], argv[2], atoi(argv[3]), "", numImages, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
        return 0;
    }

    //
    return -1;
}
