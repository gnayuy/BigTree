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

#include "cxxopts.hpp"
#include "BigTree.h"

//
int main (int argc, const char *argv[])
{
    // Usage:
    // BigTree -i <input_DIR> -o <output_DIR> -n <Number_of_Resolutions_of_TMITREE>

    // assuming input 2D TIFF (LZW compressed) images and convert to 3D TIFF blocks
    // 3D block with 256x256x256

    //
    int nScale = 3;
    string inputDir, outputDir;
    bool usingGPU = true;

    //
    try
    {
        cxxopts::Options options(argv[0], " -i <input_DIR> -o <output_DIR> -n <Number_of_Resolutions_of_TMITREE>");
        options
                .positional_help("[optional args]")
                .show_positional_help();

        options.add_options()
                ("h,help", "BigTree Version 1.0")
                ("i,input", "Input DIR", cxxopts::value<std::string>(inputDir))
                ("o,output", "Output DIR", cxxopts::value<std::string>(outputDir))
                ("n,resolutions", "N Resolutions", cxxopts::value<int>(nScale))
                ;

        auto cmds = options.parse(argc, argv);

        if (cmds.count("help"))
        {
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }

        if (cmds.count("input"))
        {
            std::cout << " -- Input 2D TIFF Images DIR: " << cmds["input"].as<std::string>() << std::endl;
        }

        if (cmds.count("output"))
        {
            std::cout << " -- Output 3D TIFF Images DIR: " << cmds["output"].as<std::string>() << std::endl;
        }

        if (cmds.count("n"))
        {
            std::cout << " -- Convert data into " << cmds["n"].as<int>() << " scales" << std::endl;
        }

    }
    catch(const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    // BigTree
    BigTree bt(inputDir, outputDir, nScale, usingGPU);

    //
    return 0;
}
