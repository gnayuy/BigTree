// BigTree.h


#include "Image.h"

#define MAX_IMAGES_STREAM 32

// blocks in each folder
class BLOCK
{
public:
    BLOCK();
    ~BLOCK();

public:
    int findNonZeroBlocks();

public:
    int offset_V;
    int offset_H;
    vector<int> offsets_D;

    uint16 lengthFileName; // 25 len("000000/000000_000000.tif") + 1
    uint16 lengthDirName; // 21 len("000000_000000_000000") + 1

    string dirName;

    uint32 nBlocksPerDir;
    vector<string> fileNames;

    uint32 height, width, depth, color, bytesPerVoxel;

    vector<bool> nonZeroBlocks;

    bool bWrite;
};

class LAYER
{
public:
    LAYER();
    ~LAYER();

public:
    uint16 rows, cols;
    float vs_x, vs_y, vs_z; // voxel sizes
    uint32 dim_V, dim_H, dim_D;

    vector<BLOCK> blocks;
};

// blocks in each resolution
class TMITREE
{
public:
    TMITREE();
    ~TMITREE();

public:
    vector<LAYER> layers;
    float org_V, org_H, org_D; // offsets (0, 0, 0)
    axis reference_V, reference_H, reference_D; // vertical, horizonal, depth
    float mdata_version; // 2
};

//
class BigTree
{
public:
    BigTree(string inputdir, string outputdir, int scales);
    ~BigTree();

public:
    int init();
    uint8* load(long zs, long ze);
    int reformat();

    // mdata.bin
    int index();

public:
    string srcdir, dstdir;
    int resolutions;
    uint32 width, height, depth; // 3D image stacks

    set<string> input2DTIFFs;
    uint32 *n_stacks_V, *n_stacks_H, *n_stacks_D;
    uint32 ****stacks_V, ****stacks_H, ****stacks_D;
    vector<string> filePaths;
    uint32 block_width, block_height, block_depth;
    uint16 datatype;
    uint32 color;
    int *halve_pow2;
    long z_ratio, z_max_res;
    uint8 *ubuffer;
    int nbits;

    TMITREE meta;
};
