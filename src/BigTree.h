// BigTree.h

#include "Image.h"

// meta
// Folder: Y/Y_X
// File: Y_X_Z.tif

typedef map<int,string, std::greater<int> > DIRs;

// block
class Cube
{
public:
    Cube();
    ~Cube();

public:
    int offset_D; // z

    string fileName; // 000000_000000_000000.tif
    uint32 depth; // 256

    string filePath; // ourdir/RESXXxXXxXX/000000/000000_000000/000000_000000_000000.tif
    bool toBeCopied;
};

// folder
class YXFolder
{
public:
    YXFolder();
    ~YXFolder();

public:
    int offset_V; // y
    int offset_H; // x

    uint16 lengthFileName; // 25 len("000000_000000_000000.tif") + 1
    uint16 lengthDirName; // 21 len("000000/000000_000000") + 1

    string dirName; // 000000/000000_000000
    string xDirPath; // ourdir/RESXXxXXxXX/000000
    string yDirPath; // ourdir/RESXXxXXxXX/000000/000000_000000

    uint32 color, bytesPerVoxel;
    uint32 height, width; // 256x256

    uint32 ncubes; // adaptive for keep only a few cubes
    bool toBeCopied;

    map<int,Cube> cubes;
};

// layer
class Layer
{
public:
    Layer();
    ~Layer();

public:
    uint16 rows, cols; // floor(dim_V/height)+1, floor(dim_H/width)+1
    uint32 dim_V, dim_H, dim_D; // dimensions y, x, z
    float vs_x, vs_y, vs_z; // voxel sizes

    uint32 ncubes;
    string layerName; // outdir/RESXXxXXxXX

    map<string, YXFolder> yxfolders; // <dirName, YXFolder>
};

// blocks in each resolution
class TMITREE
{
public:
    TMITREE();
    ~TMITREE();

public:
    vector<Layer> layers;
    float org_V, org_H, org_D; // offsets (0, 0, 0)
    axis reference_V, reference_H, reference_D; // vertical, horizonal, depth
    float mdata_version; // 2
};

//
class BigTree
{
public:
    BigTree(string inputdir, string outputdir, int scales=3, string neuron="", int numImages=16, unsigned int bsx=256, unsigned int bsy=256, unsigned int bsz=256, int nBits=4, int outDatatype=1);
    ~BigTree();

public:
    int init();
    uint8* load(long zs, long ze, long zp);
    int reformat();

    // mdata.bin
    int index();

    // updating starting z index
    int resume();

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
    int datatype_out;

    TMITREE meta;

    int zstart, zpart;
    string config4resume, config;

    int numImagesLoaded;
};

// swc
class Point
{
public:
    Point(float a, float b, float c);
    ~Point();

public:
    float x,y,z;
};

typedef vector<Point> PointCloud;

// nodes of tree
class Block
{
public:
    Block();
    Block(string fn, long xoff, long yoff, long zoff, long sx, long sy, long sz);
    ~Block();

public:
    string filepath;
    long offset_x, offset_y, offset_z;
    long size_x, size_y, size_z;
    bool visited;
};

typedef map<long, Block> OneScaleTree; // offset_z*dimx*dimy+offset_y*dimx+offset_x
typedef vector<long> OffsetType;
typedef map<long, string> ZeroBlock;

// query interested block
class QueryAndCopy
{
public:
    QueryAndCopy(string swcfile, string inputdir, string outputdir, float ratio);
    QueryAndCopy(string inputdir);
    ~QueryAndCopy();
public:
    int readSWC(string filename, float ratio = 1.0);
    int readMetaData(string filename, bool mDataDebug=false);

    int copyblock(string srcFile, string dstFile);

    int query(float x, float y, float z);
    string getDirName(string filepath);
    int label(long index);
    long findClosest(OffsetType offsets, long idx);
    long findOffset(OffsetType offsets, long idx);

public:
    OneScaleTree tree;
    PointCloud pc;

public:
    // mdata.bin
    float org_V, org_H, org_D; // offsets (0, 0, 0)
    axis reference_V, reference_H, reference_D; // vertical, horizonal, depth
    float mdata_version; // 2

    unsigned int color, bytesPerVoxel; //
    long cubex, cubey, cubez;
    long sx, sy, sz;

    OffsetType xoff, yoff, zoff;
    ZeroBlock zeroblocks;

    Layer layer;
};

// swc reader
class CSVLine
{
public:
    string const& operator[](size_t index) const;
    size_t size() const;
    void readNextRow(istream& str);

private:
    vector<std::string>    m_data;
};

class CSVIterator
{
public:
    CSVIterator(std::istream& str);
    CSVIterator();

    // Pre Increment
    CSVIterator& operator++();
    // Post increment
    CSVIterator operator++(int);
    CSVLine const& operator*()   const;
    CSVLine const* operator->()  const;

    bool operator==(CSVIterator const& rhs);
    bool operator!=(CSVIterator const& rhs);

public:
    typedef std::input_iterator_tag     iterator_category;
    typedef CSVLine                      value_type;
    typedef std::size_t                 difference_type;
    typedef CSVLine*                     pointer;
    typedef CSVLine&                     reference;

private:
    istream* m_str;
    CSVLine m_row;
};

istream& operator>>(std::istream& str, CSVLine& data);



