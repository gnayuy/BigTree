// BigTree.cpp

#include "BigTree.h"

//
Cube::Cube()
{

}

Cube::~Cube()
{

}

//
YXFolder::YXFolder()
{
    lengthFileName = 25;
    lengthDirName = 21;
    toBeCopied = false;
}

YXFolder::~YXFolder()
{
    cubes.clear();
}

//
Layer::Layer()
{
    vs_x = 1;
    vs_y = 1;
    vs_z = 1;
}

Layer::~Layer()
{
    yxfolders.clear();
}

//
Point::Point(float a, float b, float c)
{
    x = a;
    y = b;
    z = c;
}

Point::~Point()
{

}

//
Block::Block()
{

}

Block::Block(string fn, long xoff, long yoff, long zoff, long sx, long sy, long sz)
{
    filepath = fn;
    offset_x = xoff;
    offset_y = yoff;
    offset_z = zoff;
    size_x = sx;
    size_y = sy;
    size_z = sz;
    visited = false;
}

Block::~Block()
{

}

TMITREE::TMITREE()
{
    org_V = 0;
    org_H = 0;
    org_D = 0;

    reference_V = vertical;
    reference_H = horizontal;
    reference_D = depth;

    mdata_version = 2;
}

TMITREE::~TMITREE()
{
    layers.clear();
}

//
string const& CSVLine::operator[](size_t index) const
{
    return m_data[index];
}

size_t CSVLine::size() const
{
    return m_data.size();
}

void CSVLine::readNextRow(istream& str)
{
    string line;
    getline(str, line);

    stringstream lineStream(line);
    string cell1, cell2;

    m_data.clear();

    while(getline(lineStream, cell1, ','))
    {
        stringstream cellStream(cell1);
        while(getline(cellStream, cell2, ' '))
        {
            m_data.push_back(cell2);
        }
    }

    if (!lineStream && cell1.empty() && cell2.empty())
    {
        m_data.push_back("");
    }
}

istream& operator>>(std::istream& str, CSVLine& data)
{
    data.readNextRow(str);
    return str;
}

CSVIterator::CSVIterator(std::istream& str):m_str(str.good()?&str:NULL)
{
    ++(*this);
}
CSVIterator::CSVIterator():m_str(NULL)
{

}

// Pre Increment
CSVIterator& CSVIterator::operator++()
{
    if (m_str)
    {
        if (!((*m_str) >> m_row))
        {
            m_str = NULL;
        }
    }
    return *this;
}
// Post increment
CSVIterator CSVIterator::operator++(int)
{
    CSVIterator tmp(*this);
    ++(*this);
    return tmp;
}

CSVLine const& CSVIterator::operator*()   const
{
    return m_row;
}
CSVLine const* CSVIterator::operator->()  const
{
    return &m_row;
}

bool CSVIterator::operator==(CSVIterator const& rhs)
{
    return ((this == &rhs) || ((this->m_str == NULL) && (rhs.m_str == NULL)));
}
bool CSVIterator::operator!=(CSVIterator const& rhs)
{
    return !((*this) == rhs);
}

//
QueryAndCopy::QueryAndCopy(string inputdir)
{
    // test
    readMetaData(inputdir, true);
}

QueryAndCopy::QueryAndCopy(string swcfile, string inputdir, string outputdir, float ratio)
{
    // inputdir: "xxxx/RES(123x345x456)"
    // outputdir: "yyyy/RES(123x345x456)"

    // load input .swc file and mdata.bin

    //
    readSWC(swcfile, ratio);

    //
    readMetaData(inputdir);


    // find hit & neighbor blocks

    //
    long n = pc.size();

    cout<<" ... ... consider "<<n<<" nodes in "<<tree.size()<<" blocks"<<endl;

    // test
    //    map<long, Block>::iterator it = tree.begin();
    //    while(it != tree.end())
    //    {
    //        cout<<(it++)->first<<", ";
    //    }
    //    cout<<endl;


    for(long i=0; i<n; i++)
    {
        Point p = pc[i];

        query(p.x, p.y, p.z);
    }

    // copying blocks and saving mdata.bin

    //
    DIR *outdir = opendir(outputdir.c_str());
    if(outdir == NULL)
    {
        // mkdir outdir
        if(makeDir(outputdir.c_str()))
        {
            cout<<"fail in mkdir "<<outputdir<<endl;
            return;
        }
    }
    else
    {
        closedir(outdir);
    }

    //
    map<string, YXFolder>::iterator iter = layer.yxfolders.begin();
    while(iter != layer.yxfolders.end())
    {
        YXFolder yxfolder = (iter++)->second;
        // cout<<"yxfolder.ncubes "<<yxfolder.ncubes<<endl;
        layer.yxfolders[yxfolder.dirName].ncubes = yxfolder.cubes.size();
    }

    //
    string mdatabin = outputdir + "/mdata.bin";

    struct stat info;

    // mdata.bin does not exist
    if( stat( mdatabin.c_str(), &info ) != 0 )
    {
        // save mdata.bin
        FILE *file;

        file = fopen(mdatabin.c_str(), "wb");

        fwrite(&(mdata_version), sizeof(float), 1, file);
        fwrite(&(reference_V), sizeof(axis), 1, file);
        fwrite(&(reference_H), sizeof(axis), 1, file);
        fwrite(&(reference_D), sizeof(axis), 1, file);
        fwrite(&(layer.vs_x), sizeof(float), 1, file);
        fwrite(&(layer.vs_y), sizeof(float), 1, file);
        fwrite(&(layer.vs_z), sizeof(float), 1, file);
        fwrite(&(layer.vs_x), sizeof(float), 1, file);
        fwrite(&(layer.vs_y), sizeof(float), 1, file);
        fwrite(&(layer.vs_z), sizeof(float), 1, file);
        fwrite(&(org_V), sizeof(float), 1, file);
        fwrite(&(org_H), sizeof(float), 1, file);
        fwrite(&(org_D), sizeof(float), 1, file);
        fwrite(&(layer.dim_V), sizeof(unsigned int), 1, file);
        fwrite(&(layer.dim_H), sizeof(unsigned int), 1, file);
        fwrite(&(layer.dim_D), sizeof(unsigned int), 1, file);
        fwrite(&(layer.rows), sizeof(unsigned short), 1, file); // need to be updated by hits
        fwrite(&(layer.cols), sizeof(unsigned short), 1, file); // need to be updated by hits

        //cout<<layer.yxfolders.size()<<endl;

        string dirName = "zeroblocks/zeroblock"; //

        // cout<<"layer.yxfolders.size "<<layer.yxfolders.size()<<endl;

        //
        int count = 0;
        int nyxfolders = layer.yxfolders.size();
        map<string, YXFolder>::iterator iter = layer.yxfolders.begin();
        while(iter != layer.yxfolders.end())
        {
            //cout<<"testing count "<<count<<" of "<<nyxfolders<<endl;

            //
            if(count++ >= nyxfolders)
            {
                iter++;
                continue;
            }

            //cout<<"count "<<count<<" >= "<<nyxfolders<<endl;

            //
            YXFolder yxfolder = (iter++)->second;

            // cout<<"check ncubes ... "<<yxfolder.ncubes<<" at "<<count<<endl;

            // cout<<"check "<<layer.yxfolders[yxfolder.dirName].toBeCopied<<" "<<yxfolder.dirName<<endl;

            if(yxfolder.toBeCopied==false)
            {
                // continue;

                // trick: create a "zeroblocks" folder holds blocks with zeros
                // createDir(outputdir, dirName);

                //
                fwrite(&(yxfolder.height), sizeof(unsigned int), 1, file);
                fwrite(&(yxfolder.width), sizeof(unsigned int), 1, file);
                fwrite(&(layer.dim_D), sizeof(unsigned int), 1, file); // depth of all blocks
                fwrite(&(yxfolder.ncubes), sizeof(unsigned int), 1, file);
                fwrite(&(color), sizeof(unsigned int), 1, file);
                fwrite(&(yxfolder.offset_V), sizeof(int), 1, file);
                fwrite(&(yxfolder.offset_H), sizeof(int), 1, file);
                fwrite(&(yxfolder.lengthDirName), sizeof(unsigned short), 1, file);
                fwrite(const_cast<char *>(dirName.c_str()), yxfolder.lengthDirName, 1, file);

                //
                int countCube = 0;
                int ncubes = yxfolder.cubes.size();
                map<int, Cube>::iterator it = yxfolder.cubes.begin();
                while(it != yxfolder.cubes.end())
                {
                    if(countCube++ >= ncubes)
                    {
                        iter++;
                        continue;
                    }

                    // cout<<"countCube "<<countCube<<" >= "<<ncubes<<endl;

                    //
                    Cube cube = (it++)->second;

                    string cubeName = "NULL.tif";
                    unsigned short lengthCubeName = cubeName.length() + 1; // consider the end is '\0'

                    // cout<<"write/link ... "<<dirName<<" / "<<cubeName<<" "<<lengthCubeName<<endl;

                    //
                    fwrite(&(lengthCubeName), sizeof(unsigned short), 1, file);
                    fwrite(const_cast<char *>(cubeName.c_str()), lengthCubeName, 1, file);
                    fwrite(&(cube.depth), sizeof(unsigned int), 1, file);
                    fwrite(&(cube.offset_D), sizeof(int), 1, file);
                }
            }
            else
            {
                // cout<<"write ... "<<yxfolder.dirName<<endl;

                createDir(outputdir, yxfolder.dirName, layer.yxfolders[yxfolder.dirName].xDirPath, layer.yxfolders[yxfolder.dirName].yDirPath);

                //
                fwrite(&(yxfolder.height), sizeof(unsigned int), 1, file);
                fwrite(&(yxfolder.width), sizeof(unsigned int), 1, file);
                fwrite(&(layer.dim_D), sizeof(unsigned int), 1, file); // depth of all blocks
                fwrite(&(yxfolder.ncubes), sizeof(unsigned int), 1, file);
                fwrite(&(color), sizeof(unsigned int), 1, file);
                fwrite(&(yxfolder.offset_V), sizeof(int), 1, file);
                fwrite(&(yxfolder.offset_H), sizeof(int), 1, file);
                fwrite(&(yxfolder.lengthDirName), sizeof(unsigned short), 1, file);
                fwrite(const_cast<char *>(yxfolder.dirName.c_str()), yxfolder.lengthDirName, 1, file);

                //
                int countCube = 0;
                int ncubes = yxfolder.cubes.size();
                map<int, Cube>::iterator it = yxfolder.cubes.begin();
                while(it != yxfolder.cubes.end())
                {
                    //
                    if(countCube++ >= ncubes)
                    {
                        iter++;
                        continue;
                    }

                    // cout<<"countCube "<<countCube<<" >= "<<ncubes<<endl;

                    //
                    Cube cube = (it++)->second;

                    if(cube.toBeCopied==false)
                    {
                        string cubeName = "NULL.tif";
                        unsigned short lengthCubeName = cubeName.length();

                        //
                        fwrite(&(lengthCubeName), sizeof(unsigned short), 1, file);
                        fwrite(const_cast<char *>(cubeName.c_str()), lengthCubeName, 1, file);
                        fwrite(&(cube.depth), sizeof(unsigned int), 1, file);
                        fwrite(&(cube.offset_D), sizeof(int), 1, file);
                    }
                    else
                    {
                        //
                        //string srcFilePath = inputdir + "/" + yxfolder.dirName + "/" + cube.fileName;
                        //string dstFilePath = outputdir + "/" + yxfolder.dirName + "/" + cube.fileName;

                        char delim[] = "/";

                        char srcfn[1024];
                        strcpy(srcfn, inputdir.c_str());
                        strcat(srcfn,delim);
                        strcat(srcfn,yxfolder.dirName.c_str());
                        strcat(srcfn,delim);
                        strcat(srcfn,cube.fileName.c_str());

                        char dstfn[1024];
                        strcpy(dstfn, outputdir.c_str());
                        strcat(dstfn,delim);
                        strcat(dstfn,yxfolder.dirName.c_str());
                        strcat(dstfn,delim);
                        strcat(dstfn,cube.fileName.c_str());

                        string srcFilePath = string(srcfn);
                        string dstFilePath = string(dstfn);

                        //cout<<"copy block ... "<<srcFilePath<<" -> "<<dstFilePath<<endl;

                        copyblock(srcFilePath, dstFilePath);

                        //
                        fwrite(&(yxfolder.lengthFileName), sizeof(unsigned short), 1, file);
                        fwrite(const_cast<char *>(cube.fileName.c_str()), yxfolder.lengthFileName, 1, file);
                        fwrite(&(cube.depth), sizeof(unsigned int), 1, file);
                        fwrite(&(cube.offset_D), sizeof(int), 1, file);
                    }
                }
            }
            fwrite(&(bytesPerVoxel), sizeof(unsigned int), 1, file);
        }
        fclose(file);
    }
}

QueryAndCopy::~QueryAndCopy()
{

}

int QueryAndCopy::copyblock(string srcFile, string dstFile)
{
    std::ifstream  src(srcFile.c_str(), std::ios_base::in | std::ios_base::binary);

    if(src.is_open())
    {
        std::ofstream  dst(dstFile.c_str(), std::ios_base::out | std::ios_base::binary);

        if(dst.is_open())
        {
            dst << src.rdbuf();

            if(dst.bad())
            {
                cout<<"Error writing file "<<dstFile<<endl;
            }
        }
        else
        {
            cout<<"Error opening file "<<dstFile<<endl;
        }

        dst.close();
    }
    else
    {
        cout<<"Error opening file "<<srcFile<<endl;
    }

    src.close();

    //
    return 0;
}

int QueryAndCopy::readSWC(string filename, float ratio)
{
    // SWC: #n,type,x,y,z,radius,parent
    //       0, 1,  2,3,4, 5,    6

    //
    if(ratio==0)
    {
        cout<<"Invalid ratio (=0)!"<<endl;
        return -1;
    }

    //
    ifstream file(filename.c_str());

    for(CSVIterator loop(file); loop != CSVIterator(); ++loop)
    {
        size_t found = (*loop)[0].find("#");
        if(found!=std::string::npos)
        {
            if(found==0)
            {
                // skip comments
                continue;
            }
        }

        //
        Point p(atof(((*loop)[2]).c_str())/ratio, atof(((*loop)[3]).c_str())/ratio, atof(((*loop)[4]).c_str())/ratio);
        pc.push_back(p);

    }

    //
    return 0;
}

int QueryAndCopy::readMetaData(string filename, bool mDataDebug)
{
    //
    string inputdir = filename;

    DIR *outdir = opendir(inputdir.c_str());
    if(outdir == NULL)
    {
        cout<<"Empty folder: "<<inputdir<<endl;
        return -1;
    }
    else
    {
        closedir(outdir);
    }

    //
    string blockNamePrefix = inputdir + "/";

    //
    filename = inputdir + "/mdata.bin";

    struct stat info;

    // mdata.bin does not exist
    if( stat( filename.c_str(), &info ) != 0 )
    {
        cout<<filename<<" does not exist"<<endl;
        return -1;
    }
    else
    {
        // read
        FILE *file;

        file = fopen(filename.c_str(), "rb");

        fread(&(mdata_version), sizeof(float), 1, file);
        fread(&(reference_V), sizeof(axis), 1, file); // int
        fread(&(reference_H), sizeof(axis), 1, file);
        fread(&(reference_D), sizeof(axis), 1, file);
        fread(&(layer.vs_x), sizeof(float), 1, file);
        fread(&(layer.vs_y), sizeof(float), 1, file);
        fread(&(layer.vs_z), sizeof(float), 1, file);
        fread(&(layer.vs_x), sizeof(float), 1, file);
        fread(&(layer.vs_y), sizeof(float), 1, file);
        fread(&(layer.vs_z), sizeof(float), 1, file);
        fread(&(org_V), sizeof(float), 1, file);
        fread(&(org_H), sizeof(float), 1, file);
        fread(&(org_D), sizeof(float), 1, file);
        fread(&(layer.dim_V), sizeof(unsigned int), 1, file);
        fread(&(layer.dim_H), sizeof(unsigned int), 1, file);
        fread(&(layer.dim_D), sizeof(unsigned int), 1, file);
        fread(&(layer.rows), sizeof(unsigned short), 1, file);
        fread(&(layer.cols), sizeof(unsigned short), 1, file);

        sx = layer.dim_H;
        sy = layer.dim_V;
        sz = layer.dim_D;

        int count=0; // get cube size

        //
        if(mDataDebug)
        {
            cout<<"filename "<<filename<<endl;

            cout<<"meta.mdata_version "<<mdata_version<<endl;
            cout<<"meta.reference_V "<<reference_V<<endl;
            cout<<"meta.reference_H "<<reference_H<<endl;
            cout<<"meta.reference_D "<<reference_D<<endl;
            cout<<"layer.vs_x "<<layer.vs_x<<endl;
            cout<<"layer.vs_y "<<layer.vs_y<<endl;
            cout<<"layer.vs_z "<<layer.vs_z<<endl;
            cout<<"meta.org_V "<<org_V<<endl;
            cout<<"meta.org_H "<<org_H<<endl;
            cout<<"meta.org_D "<<org_D<<endl;
            cout<<"layer.dim_V "<<layer.dim_V<<endl;
            cout<<"layer.dim_H "<<layer.dim_H<<endl;
            cout<<"layer.dim_D "<<layer.dim_D<<endl;
            cout<<"layer.rows "<<layer.rows<<endl;
            cout<<"layer.cols "<<layer.cols<<endl;
        }

        //
        int n = layer.rows*layer.cols;
        for(int i=0; i<n; i++)
        {
            //
            YXFolder yxfolder;

            // char dirName[100]; // 21

            //
            fread(&(yxfolder.height), sizeof(unsigned int), 1, file);
            fread(&(yxfolder.width), sizeof(unsigned int), 1, file);
            fread(&(layer.dim_D), sizeof(unsigned int), 1, file);
            fread(&(yxfolder.ncubes), sizeof(unsigned int), 1, file);
            fread(&(color), sizeof(unsigned int), 1, file);
            fread(&(yxfolder.offset_V), sizeof(int), 1, file);
            fread(&(yxfolder.offset_H), sizeof(int), 1, file);
            fread(&(yxfolder.lengthDirName), sizeof(unsigned short), 1, file);

            string dirName(yxfolder.lengthDirName, '\0');

            fread(&(dirName[0]), sizeof(char), yxfolder.lengthDirName, file);

            yxfolder.dirName = dirName;

            //
            if(mDataDebug)
            {
                cout<<"... "<<endl;
                cout<<"HEIGHT "<<yxfolder.height<<endl;
                cout<<"WIDTH "<<yxfolder.width<<endl;
                cout<<"DEPTH "<<layer.dim_D<<endl;
                cout<<"N_BLOCKS "<<yxfolder.ncubes<<endl;
                cout<<"N_CHANS "<<color<<endl;
                cout<<"ABS_V "<<yxfolder.offset_V<<endl;
                cout<<"ABS_H "<<yxfolder.offset_H<<endl;
                cout<<"str_size "<<yxfolder.lengthDirName<<endl;
                cout<<"DIR_NAME "<<yxfolder.dirName<<endl;
                // printf("DIR_NAME: %s\n",yxfolder.dirName.c_str());
            }

            //
            for(uint32 j=0; j<yxfolder.ncubes; j++)
            {
                //
                Cube cube;

                // char fileName[100]; // 25

                //
                fread(&(yxfolder.lengthFileName), sizeof(unsigned short), 1, file);

                string fileName(yxfolder.lengthFileName, '\0');

                fread(&(fileName[0]), sizeof(char), yxfolder.lengthFileName, file);
                fread(&(cube.depth), sizeof(unsigned int), 1, file);
                fread(&(cube.offset_D), sizeof(int), 1, file);

                cube.fileName = fileName;

                yxfolder.cubes.insert(make_pair(cube.offset_D, cube));

                //
                Block block(blockNamePrefix + yxfolder.dirName + "/" + cube.fileName,
                            long(yxfolder.offset_H), long(yxfolder.offset_V), long(cube.offset_D),
                            long(yxfolder.width), long(yxfolder.height), long(cube.depth) );

                if(count==0)
                {
                    cubex = yxfolder.width;
                    cubey = yxfolder.height;
                    cubez = cube.depth;
                    count++;
                }

                if(find(xoff.begin(), xoff.end(), long(block.offset_x)) == xoff.end())
                {
                    xoff.push_back(long(block.offset_x));
                }

                if(find(yoff.begin(), yoff.end(), long(block.offset_y)) == yoff.end())
                {
                    yoff.push_back(long(block.offset_y));
                }

                if(find(zoff.begin(), zoff.end(), long(block.offset_z)) == zoff.end())
                {
                    zoff.push_back(long(block.offset_z));
                }

                tree.insert(make_pair(long(block.offset_z)*sx*sy+long(block.offset_y)*sx+long(block.offset_x), block));

                //
                if(mDataDebug)
                {
                    cout<<"... ..."<<endl;
                    cout<<"str_size "<<yxfolder.lengthFileName<<endl;
                    cout<<"FILENAMES["<<cube.offset_D<<"] "<<cube.fileName<<endl;
                    cout<<"BLOCK_SIZE+i "<<cube.depth<<endl;
                    cout<<"BLOCK_ABS_D+i "<<cube.offset_D<<endl;
                }
            }
            fread(&(bytesPerVoxel), sizeof(unsigned int), 1, file);

            if(mDataDebug)
            {
                cout<<"N_BYTESxCHAN "<<bytesPerVoxel<<endl;
            }

            layer.yxfolders.insert(make_pair(yxfolder.dirName, yxfolder));
        }
        fclose(file);
    }

    //
    return 0;
}

int QueryAndCopy::query(float x, float y, float z)
{
    //cout<<"query "<<x<<" "<<y<<" "<<z<<endl;
    //cout<<"size "<<sx<<" "<<sy<<" "<<sz<<endl;
    //cout<<"cube size "<<cubex<<" "<<cubey<<" "<<cubez<<endl;
    //cout<<"search in "<<tree.size()<<" blocks"<<endl;

    // find hit block and 6 neighbors
    if(tree.size()>0)
    {
        //        long nx = long(x)/cubex;
        //        long ny = long(y)/cubey;
        //        long nz = long(z)/cubez;

        // hit block

        //        long lx = nx*cubex;
        //        long ly = ny*cubey;
        //        long lz = nz*cubez;

        long lx = findOffset(xoff, long(x));
        long ly = findOffset(yoff, long(y));
        long lz = findOffset(zoff, long(z));

        long olx = lx;
        long oly = ly;
        //        long olz = lz;

        long index = lz*sx*sy + ly*sx + lx;

        //cout<<"node's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;


        // test
        //        map<long, Block>::iterator it = tree.begin();
        //        while(it != tree.end())
        //        {
        //            cout<<(it++)->first<<", ";
        //        }
        //        cout<<endl;

        //
        label(index);

        // 6 neighbors

        // x-
        //        if(nx-1>0)
        //        {
        //            lx = (nx - 1) * cubex;
        //            index = lz*sx*sy + ly*sx + lx;

        //            label(index);
        //        }

        lx = findOffset(xoff, long(x-cubex));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's x- neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;

        // x+
        //        lx = (nx + 1) * cubex;

        lx = findOffset(xoff, long(x+cubex));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's x+ neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;

        lx = olx;

        // y-
        //        lx = nx*cubex;

        //        if(ny-1>0)
        //        {
        //            ly = (ny - 1)*cubey;

        //            index = lz*sx*sy + ly*sx + lx;

        //            label(index);
        //        }

        ly = findOffset(yoff, long(y-cubey));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's y- neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;

        // y+
        //        ly = (ny + 1)*cubey;

        ly = findOffset(yoff, long(y+cubey));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's y+ neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;

        ly = oly;

        // z-
        //        ly = ny*cubey;

        //        if(nz-1>0)
        //        {
        //            lz = (nz - 1)*cubez;

        //            index = lz*sx*sy + ly*sx + lx;

        //            label(index);
        //        }

        lz = findOffset(yoff, long(z-cubez));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's z- neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;

        // z+
        //        lz = (nz + 1)*cubez;

        lz = findOffset(yoff, long(z+cubez));
        index = lz*sx*sy + ly*sx + lx;
        label(index);

        //cout<<"node's z+ neighbor's index "<<lx<<" "<<ly<<" "<<lz<<" "<<index<<endl;
    }
    else
    {
        cout<<"Read mdata.bin then query."<<endl;
        return -1;
    }

    //
    return 0;
}

string QueryAndCopy::getDirName(string filepath)
{
    // filepath: xxxx/RES(123x456x789)/000/000_000/000_000_000.tif
    // dirName: 000/000_000
    // -------- splits[n-3] + "/" + splits[n-2]

    vector<string> splits = splitFilePath(filepath);

    //
    size_t n = splits.size();

    if(n<3)
    {
        cout<<"Invalid filepath "<<filepath<<endl;
        return "";
    }

    string dirName = splits[n-3] + "/" + splits[n-2];

    //
    return dirName;
}

int QueryAndCopy::label(long index)
{
    //
    if(tree.find(index) != tree.end())
    {
        Block block = tree[index];

        if(block.visited == false)
        {
            //cout<<"hits the block "<<block.filepath<<" "<<block.offset_x<<" "<<block.offset_y<<" "<<block.offset_z<<" "<<index<<endl;

            string dirName = getDirName(block.filepath);

            //cout<<"check dirName: "<<dirName<<endl;

            layer.yxfolders[dirName].cubes[block.offset_z].toBeCopied = true;
            layer.yxfolders[dirName].toBeCopied = true;
            tree[index].visited = true;
        }
    }

    return 0;
}

long QueryAndCopy::findClosest(OffsetType offsets, long idx)
{
    long n = offsets.size();
    long thresh = 5;

    //
    if(n<1)
    {
        cout<<"Invalid offsets/index"<<endl;
        return -1;
    }
    else
    {
        // test
        cout<<"... offset ... ";
        for(int i=0; i<n; i++)
        {
            cout<<offsets[i]<<" ";
        }
        cout<<endl;
    }

    //
    if(idx<0)
    {
        idx = 0;
    }

    //
    long mindist = abs(idx - offsets[0]);

    long offset = offsets[0];

    if(mindist<thresh)
        return offset;

    //
    for(long i=1; i<offsets.size(); i++)
    {
        long dist = abs(idx - offsets[i]);

        if(dist<mindist)
        {
            mindist = dist;
            offset = offsets[i];

            if(mindist<thresh)
                return offset;
        }
    }

    return offset;
}

long QueryAndCopy::findOffset(OffsetType offsets, long idx)
{
    long n = offsets.size();

    //
    if(n<1)
    {
        cout<<"Invalid offsets/index"<<endl;
        return -1;
    }

    //
    if(idx<0)
    {
        idx = 0;
    }

    //
    long mindist = abs(idx - offsets[0]);

    long offset = offsets[0];
    size_t index = 0;

    //
    for(long i=1; i<offsets.size(); i++)
    {
        long dist = abs(idx - offsets[i]);

        if(dist<mindist)
        {
            mindist = dist;
            offset = offsets[i];
            index = i;
        }
    }

    if(idx<offsets[index])
        offset = offsets[index-1];

    return offset;
}

//
BigTree::BigTree(string inputdir, string outputdir, int scales, string neuron, int numImages, unsigned int bsx, unsigned int bsy, unsigned int bsz)
{
    // init
    halve_pow2 = NULL;
    n_stacks_V = NULL;
    n_stacks_H = NULL;
    n_stacks_D = NULL;

    stacks_V = NULL;
    stacks_H = NULL;
    stacks_D = NULL;

    // default parameters settings
    block_width = bsx;
    block_height = bsy;
    block_depth = bsz;

    zstart = 0;
    zpart = 1;

    config4resume = outputdir + "/.BigTreeResume.conf";
    config = outputdir + "/.BigTree.conf";

    resume();

    //
    nbits = 4; // remove lower n bits?

    numImagesLoaded = numImages;

    //
    ubuffer = NULL;

    // inputs
    resolutions = scales;

    if(resolutions<1)
    {
        cout<<"Invalide resolutions setting \n";
    }

    //
    srcdir.assign(inputdir);
    dstdir.assign(outputdir);

    //
    omp_set_num_threads(omp_get_max_threads());

    //
    if(neuron.empty())
    {
        // convert the whole dataset

        //
        if(init())
        {
            cout<<"fail in init() \n";
            exit(0);
        }

        //
        if(reformat())
        {
            cout<<"fail in reformat() \n";
            exit(0);
        }

        //
        if(index())
        {
            cout<<"fail in index() \n";
            exit(0);
        }
    }
    else
    {
        // big geometric tree

        // inputdir: "xxxx/RES(123x345x456)"
        // outputdir: "yyyy/RES(123x345x456)"

        // load input swc file "neuron" and mdata.bin

        DIRs subDirs;

        DIR *dir = opendir(srcdir.c_str());
        struct dirent *entry = readdir(dir);

        while (entry != NULL)
        {
            //if (entry->d_type == DT_DIR)
            //{
                string folderPrefix = "RES";
                string subfolder = entry->d_name;

                if(subfolder.substr(0, folderPrefix.size())==folderPrefix)
                {
                    size_t yDimStart = subfolder.find("(");
                    size_t yDimEnd = subfolder.find("x");

                    string yDim = subfolder.substr(yDimStart+1, yDimEnd - yDimStart - 1);
                    int yDimNum = atoi(yDim.c_str());

                    subDirs[ yDimNum ] = subfolder;
                }
            //}
            entry = readdir(dir);
        }
        closedir(dir);

        //
        int n=0;
        for (map<int, string>::iterator i = subDirs.begin(); i != subDirs.end(); i++)
        {
            cout << "Creating BigTree at scale "<< n << " " << i->second << endl;

            float ratio = pow(2.0, n++);
            QueryAndCopy qc(neuron, srcdir+"/"+i->second, dstdir+"/"+i->second, ratio);
        }
    }
}

BigTree::~BigTree()
{
    //
    input2DTIFFs.clear();
    filePaths.clear();

    //
    del1dp(halve_pow2);

    //
    if(stacks_V || stacks_H || stacks_D)
    {
        for(int res_i=0; res_i< resolutions; res_i++)
        {
            for(int stack_row = 0; stack_row < n_stacks_V[res_i]; stack_row++)
            {
                for(int stack_col = 0; stack_col < n_stacks_H[res_i]; stack_col++)
                {
                    delete [](stacks_V[res_i][stack_row][stack_col]);
                    delete [](stacks_H[res_i][stack_row][stack_col]);
                    delete [](stacks_D[res_i][stack_row][stack_col]);
                }
                delete [](stacks_V[res_i][stack_row]);
                delete [](stacks_H[res_i][stack_row]);
                delete [](stacks_D[res_i][stack_row]);
            }
            delete [](stacks_V[res_i]);
            delete [](stacks_H[res_i]);
            delete [](stacks_D[res_i]);
        }
        delete []stacks_V;
        delete []stacks_H;
        delete []stacks_D;
    }

    //
    del1dp(n_stacks_V);
    del1dp(n_stacks_H);
    del1dp(n_stacks_D);
}

int BigTree::init()
{
    //
    DIR *indir = opendir(srcdir.c_str());
    if(indir == NULL)
    {
        cout<< srcdir <<": No such file or directory"  << endl;
        closedir(indir);
        return -1;
    }
    else
    {
        // get image list
        struct dirent *dirinfo = readdir(indir);
        while(dirinfo)
        {
            if(!strcmp(dirinfo->d_name,".") || !strcmp(dirinfo->d_name,".."))
            {
                dirinfo = readdir(indir);
                continue;
            }
            input2DTIFFs.insert(srcdir + "/" + dirinfo->d_name); // absolute path

            dirinfo = readdir(indir);
        }
        closedir(indir);
    }

    if(input2DTIFFs.size()<1)
    {
        cout<<"No TIFF file found \n";
        return -1;
    }

    //
    string firstfilepath = *input2DTIFFs.begin();
    loadTiffMetaInfo(const_cast<char*>(firstfilepath.c_str()), width, height, depth, color, datatype);
    depth = input2DTIFFs.size();

    cout<<"Image Info obtained from "<<firstfilepath<<endl;
    cout<<"Image Size "<<width<<"x"<<height<<"x"<<depth<<"x"<<color<<" with "<<datatype<<endl;

    // for the case with small z
    float w = width;
    float h = height;
    float d = depth;
    long n = 1;
    for(size_t i=0; i<resolutions; i++)
    {
        w *= 0.5;
        h *= 0.5;
        d *= 0.5;

        if(w>=1 && h>=1 && d>=1)
        {
            n++;
        }
        else
        {
            break;
        }
    }

    if(n<resolutions)
        resolutions = n;

    cout<<"resolutions "<<resolutions<<endl;

    try
    {
        halve_pow2 = new int [resolutions];
    }
    catch(...)
    {
        cout<<"fail to alloc memory for halve_pow2"<<endl;
        return -1;
    }
    for(int i=0; i<resolutions; i++ )
    {
        halve_pow2[i] = i;
    }

    DIR *outdir = opendir(dstdir.c_str());
    if(outdir == NULL)
    {
        // mkdir outdir
        if(makeDir(dstdir.c_str()))
        {
            cout<<"fail in mkdir "<<dstdir<<endl;
            return -1;
        }
    }

    // Make Hierarchical Dirs
    try
    {
        n_stacks_V = new uint32 [resolutions];
        n_stacks_H = new uint32 [resolutions];
        n_stacks_D = new uint32 [resolutions];

        stacks_V = new uint32 ***[resolutions];
        stacks_H = new uint32 ***[resolutions];
        stacks_D = new uint32 ***[resolutions];
    }
    catch(...)
    {
        cout<<"fail to alloc memory for stacks' resolutions info"<<endl;
        return -1;
    }

    filePaths.clear();

    //
    for(int res_i=0; res_i< resolutions; res_i++)
    {
        n_stacks_V[res_i] = (int) ceil ( (height/pow(2,res_i)) / (float) block_height );
        n_stacks_H[res_i] = (int) ceil ( (width/pow(2,res_i))  / (float) block_width  );
        n_stacks_D[res_i] = (int) ceil ( (depth/pow(2,halve_pow2[res_i]))  / (float) block_depth  );

        //cout<<"n_stacks_D["<<res_i<<"] "<<n_stacks_D[res_i]<<endl;

        try
        {
            stacks_V[res_i] = new uint32 **[n_stacks_V[res_i]];
            stacks_H[res_i]  = new uint32 **[n_stacks_V[res_i]];
            stacks_D[res_i]  = new uint32 **[n_stacks_V[res_i]];
        }
        catch(...)
        {
            cout<<"fail to alloc memory"<<endl;
            return -1;
        }

        //
        for(long stack_row = 0; stack_row < n_stacks_V[res_i]; stack_row++)
        {
            try
            {
                stacks_V[res_i][stack_row] = new uint32 *[n_stacks_H[res_i]];
                stacks_H[res_i][stack_row] = new uint32 *[n_stacks_H[res_i]];
                stacks_D[res_i][stack_row] = new uint32 *[n_stacks_H[res_i]];
            }
            catch(...)
            {
                cout<<"fail to alloc memory"<<endl;
                return -1;
            }

            //
            for(long stack_col = 0; stack_col < n_stacks_H[res_i]; stack_col++)
            {
                try
                {
                    stacks_V[res_i][stack_row][stack_col] = new uint32[n_stacks_D[res_i]];
                    stacks_H[res_i][stack_row][stack_col] = new uint32[n_stacks_D[res_i]];
                    stacks_D[res_i][stack_row][stack_col] = new uint32[n_stacks_D[res_i]];
                }
                catch(...)
                {
                    cout<<"fail to alloc memory"<<endl;
                    return -1;
                }

                //
                for(long stack_sli = 0; stack_sli < n_stacks_D[res_i]; stack_sli++)
                {
                    stacks_V[res_i][stack_row][stack_col][stack_sli] =
                            ((uint32)(height/pow(2,res_i))) / n_stacks_V[res_i] + (stack_row < ((int)(height/pow(2,res_i))) % n_stacks_V[res_i] ? 1:0);
                    stacks_H[res_i][stack_row][stack_col][stack_sli] =
                            ((uint32)(width/pow(2,res_i)))  / n_stacks_H[res_i] + (stack_col < ((int)(width/pow(2,res_i)))  % n_stacks_H[res_i] ? 1:0);
                    stacks_D[res_i][stack_row][stack_col][stack_sli] =
                            ((uint32)(depth/pow(2,halve_pow2[res_i])))  / n_stacks_D[res_i] + (stack_sli < ((int)(depth/pow(2,halve_pow2[res_i]))) % n_stacks_D[res_i] ? 1:0);
                }
            }
        }

        //
        stringstream filepath;
        filepath << dstdir<<"/RES("<<(uint32)(height/pow(2,res_i))<<"x"<<(uint32)(width/pow(2,res_i))<<"x"<<(uint32)(depth/pow(2,halve_pow2[res_i]))<<")";

        //
        filePaths.push_back(filepath.str());

        //
        if(makeDir(filepath.str().c_str()))
        {
            cout<<"fail in mkdir "<<filepath.str()<<endl;
            return -1;
        }
    }

    //
    if(outdir)
    {
        closedir(outdir);
    }

    //
    z_max_res = max(min(numImagesLoaded,(int)block_depth/2),(int)pow(2,halve_pow2[resolutions-1]));
    if ( (z_max_res > 1) && z_max_res > block_depth/2 )
    {
        cout<<"too many resolutions "<<resolutions<<endl;
        return -1;
    }
    z_ratio=depth/z_max_res;

    //
    return 0;
}

uint8 *BigTree::load(long zs, long ze, long zp)
{
    // resume
    ofstream outfile;
    outfile.open(config4resume.c_str());

    outfile << zs << endl;
    outfile << zp << endl;

    outfile.close();

    //
    long sbv_V, sbv_H, sbv_D;

    sbv_V = height;
    sbv_H = width;
    sbv_D = ze - zs;

    z_ratio=sbv_D/z_max_res;

    //
    uint8 *subvol = NULL;

    try
    {
        subvol = new uint8 [sbv_V * sbv_H * sbv_D * datatype];
    }
    catch(...)
    {
        cout<<"failed to alloc memory for subvol \n";
        return NULL;
    }

    // fstream TIFFs from disk to memory
    vector<stringstream*> dataInMemory;
    vector<uint8*> imgList;

    //
    int k;
    for(k=0; k<sbv_D; k++)
    {
        //building image path
        string slicepath = *next(input2DTIFFs.begin(), zs + k);

        cout<<"load ... "<<slicepath<<endl;

        //
        ifstream inFile;
        inFile.open(slicepath.c_str());
        if (!inFile) {
            cerr << "Unable to open file "<<slicepath<<endl;
            return NULL;
        }

        //
        dataInMemory.push_back(new stringstream);
        *dataInMemory[k] << inFile.rdbuf();

        //
        inFile.close();

        //
        uint8 *slice = subvol + (k*sbv_V*sbv_H*datatype);
        imgList.push_back(slice);
    }

    // multithreaded read TIFFs from memory
    #pragma omp parallel
    {
        #pragma omp for
        for(k=0; k<sbv_D; k++)
        {
            // unsigned int sx, sy;
            readTiff(dataInMemory[k],imgList[k],0,0,0,sbv_V-1,0,sbv_H-1);
        }
    }

    //
    return subvol;
}

int BigTree::reformat()
{
    // meta
    //    vector<LAYER> layers;

    //
    int stack_block[TMITREE_MAX_HEIGHT];
    int slice_start[TMITREE_MAX_HEIGHT];
    int slice_end[TMITREE_MAX_HEIGHT];

    int nzsize[TMITREE_MAX_HEIGHT];

    for(int res_i=0; res_i< resolutions; res_i++)
    {
        stack_block[res_i] = 0;
        slice_start[res_i] = 0;
        slice_end[res_i] = slice_start[res_i] + stacks_D[res_i][0][0][0] - 1;

        //cout<<"slice_end["<<res_i<<"] "<<slice_end[res_i]<<endl;

        nzsize[res_i] = 0;
    }

    //
    for(long z=zstart, z_parts=zpart; z<depth; z+=z_max_res, z_parts++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ubuffer = load(z,(z+z_max_res <= depth) ? (z+z_max_res) : depth, z_parts);

        auto end = std::chrono::high_resolution_clock::now();

        cout<<"load a sub volume takes "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<" ms."<<endl;

        // remove lower 4 bits for 16bit input data
        if(datatype>1 && nbits)
        {
            long totalvoxels = (height * width * ((z_ratio>0) ? z_max_res : (depth%z_max_res)))*color;
            if ( datatype == 2 )
            {
                #pragma omp parallel
                {
                    uint16 *ptr = (uint16 *) ubuffer;
                    #pragma omp for
                    for(long i=0; i<totalvoxels; i++ )
                    {
                        // ptr[i] = ptr[i] >> nbits << nbits; // 16-bit
                        ptr[i] = ptr[i] >> nbits;
                    }
                }
            }
        }

        // saving the sub volume
        start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<resolutions; i++)
        {
            //cout<<"resolution "<<i<<endl;

            // meta
            Layer layer;
            layer.rows = n_stacks_V[i];
            layer.cols = n_stacks_H[i];

            layer.vs_x = pow(2, i);
            layer.vs_y = pow(2, i);
            layer.vs_z = pow(2, halve_pow2[i]);

            layer.dim_V = (uint32)(height/layer.vs_y);
            layer.dim_H = (uint32)(width/layer.vs_x);
            layer.dim_D = (uint32)(depth/layer.vs_z);

            //
            long nCopies = 0;
            string srcFile;

            // check if current block is changed
            if ( (z / pow(2,halve_pow2[i])) > slice_end[i] ) {
                stack_block[i]++;
                slice_start[i] = slice_end[i] + 1;
                slice_end[i] += stacks_D[i][0][0][stack_block[i]];
            }

            // find abs_pos_z at resolution i
            std::stringstream abs_pos_z;
            abs_pos_z.width(6);
            abs_pos_z.fill('0');
            abs_pos_z << (int)((pow(2,halve_pow2[i])*slice_start[i]) * 10);

            // compute the number of slice of previous groups at resolution i
            // note that z_parts in the number and not an index (starts from 1)
            long n_slices_pred  = (z_parts - 1) * z_max_res / pow(2,halve_pow2[i]);

            // buffer size along D is different when the remainder of the subdivision by z_max_res is considered
            long z_size = (z_ratio>0) ? z_max_res : (depth%z_max_res);

            //cout<<"z_parts "<<z_parts<<" z_ratio "<<z_ratio<<" z_max_res "<<z_max_res<<" depth "<<depth<<endl;

            //halvesampling resolution if current resolution is not the deepest one
            if(i!=0)
            {
                if ( halve_pow2[i] == (halve_pow2[i-1]+1) )
                {
                    //cout<<"3D downsampling \n";

                    // 3D
                    halveSample(ubuffer,(int)height/(pow(2,i-1)),(int)width/(pow(2,i-1)),(int)z_size/(pow(2,halve_pow2[i-1])),HALVE_BY_MAX,datatype);

                    // debug
                    // writeTiff3DFile("test.tif", ubuffer, (int)width/(pow(2,i)), (int)height/(pow(2,i)), (int)z_size/(pow(2,halve_pow2[i])), 1, datatype);
                }
                else if ( halve_pow2[i] == halve_pow2[i-1] )
                {
                    //cout<<"2D downsampling \n";

                    // 2D
                    halveSample2D(ubuffer,(int)height/(pow(2,i-1)),(int)width/(pow(2,i-1)),(int)z_size/(pow(2,halve_pow2[i-1])),HALVE_BY_MAX,datatype);
                }
                else
                {
                    cout<<"halve sampling level "<<halve_pow2[i]<<" not supported at resolution "<<i<<endl;
                    return -1;
                }
            }

            // saving at current resolution if it has been selected and iff buffer is at least 1 voxel (Z) deep
            if((z_size/(pow(2,halve_pow2[i]))) > 0)
            {
                //storing in 'base_path' the absolute path of the directory that will contain all stacks
                std::stringstream base_path;
                base_path << filePaths[i] << "/";

                //cout<<"base_path "<<base_path.str()<<endl;

                //looping on new stacks
                for(int stack_row = 0, start_height = 0, end_height = 0; stack_row < n_stacks_V[i]; stack_row++)
                {
                    //incrementing end_height
                    end_height = start_height + stacks_V[i][stack_row][0][0]-1;

                    //computing V_DIR_path and creating the directory the first time it is needed
                    std::stringstream multires_merging_x_pos;
                    multires_merging_x_pos.width(6);
                    multires_merging_x_pos.fill('0');
                    multires_merging_x_pos << start_height*(int)pow(2.0,i) * 10;

                    std::stringstream V_DIR_path;
                    V_DIR_path << base_path.str() << multires_merging_x_pos.str();

                    //cout<<"V_DIR_path "<<V_DIR_path.str()<<endl;

                    if(z==0)
                    {
                        if(makeDir(V_DIR_path.str().c_str()))
                        {
                            cout<<" unable to create V_DIR "<<V_DIR_path.str()<<endl;
                            return -1;
                        }
                    }

                    unsigned int sz[4];
                    // int datatype_out = 2; // changed to 16-bit 6/1/2018 yy
                    int datatype_out = 1;

                    //
                    for(int stack_column = 0, start_width=0, end_width=0; stack_column < n_stacks_H[i]; stack_column++)
                    {
                        //
                        end_width  = start_width  + stacks_H[i][stack_row][stack_column][0]-1;

                        //computing H_DIR_path and creating the directory the first time it is needed
                        std::stringstream multires_merging_y_pos;
                        multires_merging_y_pos.width(6);
                        multires_merging_y_pos.fill('0');
                        multires_merging_y_pos << start_width*(int)pow(2.0,i) * 10;

                        std::stringstream H_DIR_path;
                        H_DIR_path << V_DIR_path.str() << "/" << multires_merging_x_pos.str() << "_" << multires_merging_y_pos.str();

                        //cout<<"H_DIR_path "<<H_DIR_path.str()<<endl;

                        // meta
                        YXFolder yxfolder;

                        //
                        if(z==0)
                        {
                            if(makeDir(H_DIR_path.str().c_str()))
                            {
                                cout<<" unable to create H_DIR "<<H_DIR_path.str()<<endl;
                                return -1;
                            }
                            else
                            {
                                // the directory has been created for the first time
                                // initialize block files
                                sz[0] = stacks_H[i][stack_row][stack_column][0];
                                sz[1] = stacks_V[i][stack_row][stack_column][0];
                                sz[3] = 1;

                                // meta
                                yxfolder.width = sz[0];
                                yxfolder.height = sz[1];
                                yxfolder.color = sz[3];
                                yxfolder.bytesPerVoxel = datatype_out;

                                yxfolder.dirName = multires_merging_x_pos.str() + "/" + multires_merging_x_pos.str() + "_" + multires_merging_y_pos.str();
                                yxfolder.offset_H = start_width;
                                yxfolder.offset_V = start_height;

                                //
                                int slice_start_temp = 0;
                                for ( int j=0; j < n_stacks_D[i]; j++ )
                                {
                                    bool copying = false;

                                    if(sz[2] == stacks_D[i][stack_row][stack_column][j])
                                        copying = true;

                                    sz[2] = stacks_D[i][stack_row][stack_column][j];

                                    //cout<<" ... "<<j<<" sz[2] "<<sz[2]<<" slice_end "<<slice_end[i]<<endl;

                                    std::stringstream abs_pos_z_temp;
                                    abs_pos_z_temp.width(6);
                                    abs_pos_z_temp.fill('0');
                                    abs_pos_z_temp << (int)((pow(2,halve_pow2[i])*slice_start_temp) * 10);

                                    std::stringstream img_path_temp;
                                    img_path_temp << H_DIR_path.str() << "/" << multires_merging_x_pos.str() << "_" << multires_merging_y_pos.str() << "_" << abs_pos_z_temp.str()<<".tif";

                                    //cout<<"when z=0: z "<<z<<" ("<<sz[0]<<", "<<sz[1]<<", "<<sz[2]<<") "<<abs_pos_z_temp.str()<<endl;

                                    //
                                    Cube cube;
                                    cube.fileName = multires_merging_x_pos.str() + "_" + multires_merging_y_pos.str() + "_" + abs_pos_z_temp.str() + ".tif";
                                    cube.filePath = img_path_temp.str();
                                    cube.depth = sz[2];

                                    if(yxfolder.cubes.size()>0)
                                    {
                                        map<int,Cube>::reverse_iterator rit;
                                        rit = yxfolder.cubes.rbegin();

                                        cube.offset_D = rit->second.offset_D + rit->second.depth;
                                    }
                                    else
                                    {
                                        cube.offset_D = 0;
                                    }
                                    yxfolder.cubes.insert(make_pair(cube.offset_D, cube));

                                    // auto start_init = std::chrono::high_resolution_clock::now();
                                    if(nCopies==0)
                                    {
                                        if(initTiff3DFile((char *)img_path_temp.str().c_str(),sz[0],sz[1],sz[2],sz[3],datatype_out) != 0)
                                        {
                                            cout<<"fail in initTiff3DFile\n";
                                            return -1;
                                        }
                                        srcFile = img_path_temp.str();
                                    }
                                    else if(copying)
                                    {
                                        copyFile(srcFile.c_str(), img_path_temp.str().c_str());
                                    }
                                    else
                                    {
                                        if(initTiff3DFile((char *)img_path_temp.str().c_str(),sz[0],sz[1],sz[2],sz[3],datatype_out) != 0)
                                        {
                                            cout<<"fail in initTiff3DFile\n";
                                            return -1;
                                        }
                                    }
                                    nCopies++;

                                    //
                                    //auto end_init = std::chrono::high_resolution_clock::now();
                                    //cout<<"writing chunk images takes "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init).count()<<" ms."<<endl;

                                    //
                                    slice_start_temp += (int)sz[2];
                                } // j
                            } // after create H_DIR
                        } // z

                        //saving HERE

                        //
                        std::stringstream partial_img_path;
                        partial_img_path << H_DIR_path.str() << "/" << multires_merging_x_pos.str() << "_" << multires_merging_y_pos.str() << "_";

                        int slice_ind = (int)(n_slices_pred - slice_start[i]);

                        std::stringstream img_path;
                        img_path << partial_img_path.str() << abs_pos_z.str() << ".tif";

                        //cout<<"img_path "<<img_path.str()<<endl;

                        //
                        void *fhandle = 0;
                        int  n_pages_block = stacks_D[i][0][0][stack_block[i]]; // number of pages of current block
                        bool block_changed = false; // true if block is changed executing the next for cycle

                        if(openTiff3DFile((char *)img_path.str().c_str(),(char *)("a"),fhandle,true))
                        {
                            cout<<"fail in openTiff3DFile"<<endl;
                            return -1;
                        }

                        //
                        sz[0] = end_width - start_width + 1;
                        sz[1] = end_height - start_height + 1;
                        sz[2] = n_pages_block;
                        sz[3] = 1;
                        long szChunk = sz[0]*sz[1]*sz[3]*datatype_out;
                        unsigned char *p = NULL;

                        try
                        {
                            p = new unsigned char [szChunk];
                            memset(p, 0, szChunk);
                        }
                        catch(...)
                        {
                            cout<<"fail to alloc memory \n";
                        }

                        bool blocksaved = false;

                        //cout<<"z "<<z<<endl;

                        // WARNING: assumes that block size along z is not less that z_size/(powInt(2,i))
                        for(int buffer_z=0; buffer_z<(int)(z_size/(pow(2,halve_pow2[i]))); buffer_z++, slice_ind++)
                        {
                            //cout<<"buffer_z "<<buffer_z<<" slice_ind "<<slice_ind<<" z "<<z<<" z_size/(pow(2,halve_pow2[i]) "<<z_size/(pow(2,halve_pow2[i]))<<" z_size "<<z_size<<endl;
                            //cout<<"(z / pow(2,halve_pow2[i]) + buffer_z) "<<z / pow(2,halve_pow2[i]) + buffer_z<<" slice_end["<<i<<"] "<<slice_end[i]<<endl;

                            // z is an absolute index in volume while slice index should be computed on a relative basis
                            if ( (int)(z / pow(2,halve_pow2[i]) + buffer_z) > slice_end[i] && !block_changed)
                            {
                                //cout<<"block changed "<<slice_end[i]<<endl;

                                // start a new block along z
                                std::stringstream abs_pos_z_next;
                                abs_pos_z_next.width(6);
                                abs_pos_z_next.fill('0');
                                abs_pos_z_next << (pow(2,halve_pow2[i])*(slice_end[i]+1)) * 10;
                                img_path.str("");
                                img_path << partial_img_path.str() << abs_pos_z_next.str() << ".tif";

                                //cout<<"... img_path "<<img_path.str()<<endl;

                                slice_ind = 0;

                                // close(fhandle) i.e. file corresponding to current block
                                TIFFClose((TIFF *) fhandle);
                                if(openTiff3DFile((char *)img_path.str().c_str(),(char *)("a"),fhandle,true))
                                {
                                    cout<<"fail in openTiff3DFile"<<endl;
                                    return -1;
                                }
                                n_pages_block = stacks_D[i][0][0][stack_block[i]+1];
                                block_changed = true;

                                sz[2] = n_pages_block;
                                szChunk = sz[0]*sz[1]*sz[3]*datatype_out;

                                //
                                if(!p)
                                {
                                    try
                                    {
                                        p = new unsigned char [szChunk];
                                        memset(p, 0, szChunk);
                                    }
                                    catch(...)
                                    {
                                        cout<<"fail to alloc memory \n";
                                    }
                                }
                                else
                                {
                                    memset(p, 0, szChunk);
                                }
                            }

                            //
                            long raw_img_width = width/(pow(2,i));

                            //
                            if(datatype == 2)
                            {
                                // 16-bit input
                                long offset = buffer_z*(long)(height/pow(2,i))*(long)(width/pow(2,i));
                                uint16 *raw_ch16 = (uint16 *) ubuffer + offset;

                                //cout<<"pointer p: "<<static_cast<void*>(p)<<endl;
                                //cout<<"pointer raw data: "<<static_cast<void*>(raw_ch16)<<endl;

                                if(datatype_out == 1)
                                {
                                    // 8-bit output

                                    //
                                    #pragma omp parallel for collapse(2)
                                    for(long y=0; y<sz[1]; y++)
                                    {
                                        for(long x=0; x<sz[0]; x++)
                                        {
                                            p[y*sz[0]+x] = raw_ch16[(y+start_height)*(raw_img_width) + (x+start_width)];
                                        }
                                    }

                                    // temporary save all the way (version 1.01 5/25/2018)
                                    //                                        int temp_n_chans = color;
                                    //                                        if(temp_n_chans==2)
                                    //                                            temp_n_chans++;

                                    appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)p,sz[0],sz[1],color,8,sz[2]);
                                    blocksaved = true;

                                    //
                                    //                                        int numNonZeros = 0;
                                    //                                        int saveVoxelThresh = 1;

                                    //                                        #pragma omp parallel for reduction(+:numNonZeros)
                                    //                                        for(int x=0; x<szChunk; x++)
                                    //                                        {
                                    //                                            if(p[x]>0)
                                    //                                                numNonZeros++;
                                    //                                        }

                                    //                                        if(numNonZeros>saveVoxelThresh)
                                    //                                        {
                                    //                                            int temp_n_chans = color;
                                    //                                            if(temp_n_chans==2)
                                    //                                                temp_n_chans++;

                                    //                                            appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)p,sz[0],sz[1],temp_n_chans,8,sz[2]);
                                    //                                            blocksaved = true;
                                    //                                        }
                                }
                                else
                                {
                                    // 16-bit output

                                    uint16 *out_ch16 = (uint16 *) p;

                                    //
                                    #pragma omp parallel for collapse(2)
                                    for(long y=0; y<sz[1]; y++)
                                    {
                                        for(long x=0; x<sz[0]; x++)
                                        {
                                            out_ch16[y*sz[0]+x] = raw_ch16[(y+start_height)*(raw_img_width) + (x+start_width)];
                                        }
                                    }

                                    // temporary save all the way (version 1.01 5/25/2018)
                                    int temp_n_chans = color;
                                    if(temp_n_chans==2)
                                        temp_n_chans++;

                                    appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)out_ch16,sz[0],sz[1],temp_n_chans,16,sz[2]);
                                    blocksaved = true;

                                }

                            }
                            else if(datatype == 1)
                            {
                                // 8-bit input
                                long offset = buffer_z*(long)(height/pow(2,i))*(long)(width/pow(2,i));
                                uint8 *raw_ch8 = (uint8 *) ubuffer + offset;

                                if(datatype_out == 1)
                                {
                                    // 8-bit output

                                    //
                                    //#pragma omp parallel for collapse(2)
                                    for(long y=0; y<sz[1]; y++)
                                    {
                                        for(long x=0; x<sz[0]; x++)
                                        {
                                            p[y*sz[0]+x] = raw_ch8[(y+start_height)*(raw_img_width) + (x+start_width)];
                                        }
                                    }

                                    // temporary save all the way (version 1.01 5/25/2018)
//                                    int temp_n_chans = color;
//                                    if(temp_n_chans==2)
//                                        temp_n_chans++;

                                    appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)p,sz[0],sz[1],color,8,sz[2]);
                                    blocksaved = true;

                                    //
                                    //                                        int numNonZeros = 0;
                                    //                                        int saveVoxelThresh = 1;

                                    //                                        #pragma omp parallel for reduction(+:numNonZeros)
                                    //                                        for(int x=0; x<szChunk; x++)
                                    //                                        {
                                    //                                            if(p[x]>0)
                                    //                                                numNonZeros++;
                                    //                                        }

                                    //                                        //cout<<"... raw_img_width "<<raw_img_width<<" offset "<<offset<<" height/pow(2,i) "<<height/pow(2,i)<<" width/pow(2,i) "<<width/pow(2,i)<<endl;

                                    //                                        if(numNonZeros>saveVoxelThresh)
                                    //                                        {
                                    //                                            int temp_n_chans = color;
                                    //                                            if(temp_n_chans==2)
                                    //                                                temp_n_chans++;

                                    //                                            //cout<<"... save slice_ind: "<<slice_ind<<endl;
                                    //                                            appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)p,sz[0],sz[1],temp_n_chans,8,sz[2]);
                                    //                                            blocksaved = true;
                                    //                                        }
                                }
                                else
                                {
                                    // 16-bit output

                                }
                            }
                            else
                            {
                                // other datatypes
                            }
                        }

                        //
                        del1dp(p);

                        // close(fhandle) i.e. currently opened file
                        TIFFClose((TIFF *) fhandle);

                        //
                        start_width  += stacks_H[i][stack_row][stack_column][0];

                        //
                        //                        if(addMeta)
                        //                        {
                        //                            block.nonZeroBlocks.push_back(blocksaved);
                        //                            block.nBlocksPerDir = block.fileNames.size();
                        //                            layer.blocks.push_back(block);
                        //                            layer.n_scale = i;
                        //                        }

                        yxfolder.ncubes = yxfolder.cubes.size();
                        layer.yxfolders.insert(make_pair(yxfolder.dirName, yxfolder));
                    }
                    start_height += stacks_V[i][stack_row][0][0];
                }
            }

            //
            //            if(!layer.blocks.empty())
            //            {
            //                layers.push_back(layer);
            //            }
            meta.layers.push_back(layer);
        }

        //releasing allocated memory
        del1dp(ubuffer);

        //
        end = std::chrono::high_resolution_clock::now();
        cout<<"writing sub volume's chunk images takes "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<" ms."<<endl;
    }

    //
    return 0;
}

int BigTree::index()
{
    // saving mdata.bin for fast indexing image blocks instead of re-scan files every time

    // voxel size 1 micron by default
    // original offsets 0 mm by default

    if(meta.layers.empty())
    {
        cout<<"Need meta data for further visualization"<<endl;
        return -1;
    }

    //
    for(int res_i=0; res_i< resolutions; res_i++)
    {
        //cout<<"res_i "<<res_i<<endl;

        //
        string filename = filePaths[res_i] + "/mdata.bin";

        //
        Layer layer = meta.layers[res_i];

        // save
        FILE *file;

        file = fopen(filename.c_str(), "w");

        fwrite(&(meta.mdata_version), sizeof(float), 1, file);
        fwrite(&(meta.reference_V), sizeof(axis), 1, file);
        fwrite(&(meta.reference_H), sizeof(axis), 1, file);
        fwrite(&(meta.reference_D), sizeof(axis), 1, file);
        fwrite(&(layer.vs_x), sizeof(float), 1, file);
        fwrite(&(layer.vs_y), sizeof(float), 1, file);
        fwrite(&(layer.vs_z), sizeof(float), 1, file);
        fwrite(&(layer.vs_x), sizeof(float), 1, file);
        fwrite(&(layer.vs_y), sizeof(float), 1, file);
        fwrite(&(layer.vs_z), sizeof(float), 1, file);
        fwrite(&(meta.org_V), sizeof(float), 1, file);
        fwrite(&(meta.org_H), sizeof(float), 1, file);
        fwrite(&(meta.org_D), sizeof(float), 1, file);
        fwrite(&(layer.dim_V), sizeof(uint32), 1, file);
        fwrite(&(layer.dim_H), sizeof(uint32), 1, file);
        fwrite(&(layer.dim_D), sizeof(uint32), 1, file);
        fwrite(&(layer.rows), sizeof(uint16), 1, file);
        fwrite(&(layer.cols), sizeof(uint16), 1, file);

//        cout<<"filename "<<filename<<endl;

//        cout<<"meta.mdata_version "<<meta.mdata_version<<endl;
//        cout<<"meta.reference_V "<<meta.reference_V<<endl;
//        cout<<"meta.reference_H "<<meta.reference_H<<endl;
//        cout<<"meta.reference_D "<<meta.reference_D<<endl;
//        cout<<"layer.vs_x "<<layer.vs_x<<endl;
//        cout<<"layer.vs_y "<<layer.vs_y<<endl;
//        cout<<"layer.vs_z "<<layer.vs_z<<endl;
//        cout<<"layer.vs_x "<<layer.vs_x<<endl;
//        cout<<"layer.vs_y "<<layer.vs_y<<endl;
//        cout<<"layer.vs_z "<<layer.vs_z<<endl;
//        cout<<"meta.org_V "<<meta.org_V<<endl;
//        cout<<"meta.org_H "<<meta.org_H<<endl;
//        cout<<"meta.org_D "<<meta.org_D<<endl;
//        cout<<"layer.dim_V "<<layer.dim_V<<endl;
//        cout<<"layer.dim_H "<<layer.dim_H<<endl;
//        cout<<"layer.dim_D "<<layer.dim_D<<endl;
//        cout<<"layer.rows "<<layer.rows<<endl;
//        cout<<"layer.cols "<<layer.cols<<endl;

        //
        int count = 0;
        int nyxfolders = layer.yxfolders.size();
        map<string, YXFolder>::iterator iter = layer.yxfolders.begin();
        while(iter != layer.yxfolders.end())
        {
            //cout<<"... count "<<count<<" of "<<nyxfolders<<endl;

            //
            if(count++ >= nyxfolders)
            {
                iter++;
                continue;
            }

            // cout<<"count "<<count<<" >= "<<nyxfolders<<endl;

            //
            YXFolder yxfolder = (iter++)->second;

            //cout<<"check ncubes ... "<<yxfolder.ncubes<<" at "<<count<<endl;

            //
            fwrite(&(yxfolder.height), sizeof(unsigned int), 1, file);
            fwrite(&(yxfolder.width), sizeof(unsigned int), 1, file);
            fwrite(&(layer.dim_D), sizeof(unsigned int), 1, file); // depth of all blocks
            fwrite(&(yxfolder.ncubes), sizeof(unsigned int), 1, file);
            fwrite(&(color), sizeof(unsigned int), 1, file);
            fwrite(&(yxfolder.offset_V), sizeof(int), 1, file);
            fwrite(&(yxfolder.offset_H), sizeof(int), 1, file);
            fwrite(&(yxfolder.lengthDirName), sizeof(unsigned short), 1, file);
            fwrite(const_cast<char *>(yxfolder.dirName.c_str()), yxfolder.lengthDirName, 1, file);

            //cout<<yxfolder.height<<" "<<yxfolder.width<<" "<<layer.dim_D<<" "<<color<<" "<<yxfolder.offset_V<<" "<<yxfolder.offset_H<<endl;

            //
            int countCube = 0;
            int ncubes = yxfolder.cubes.size();
            map<int, Cube>::iterator it = yxfolder.cubes.begin();
            while(it != yxfolder.cubes.end())
            {
                //
                if(countCube++ >= ncubes)
                {
                    iter++;
                    continue;
                }

                //cout<<"countCube "<<countCube<<" >= "<<ncubes<<endl;

                //
                Cube cube = (it++)->second;

                //
                fwrite(&(yxfolder.lengthFileName), sizeof(unsigned short), 1, file);
                fwrite(const_cast<char *>(cube.fileName.c_str()), yxfolder.lengthFileName, 1, file);
                fwrite(&(cube.depth), sizeof(unsigned int), 1, file);
                fwrite(&(cube.offset_D), sizeof(int), 1, file);

                //cout<<cube.depth<<" "<<cube.offset_D<<" "<<cube.fileName<<endl;
            }
            fwrite(&(yxfolder.bytesPerVoxel), sizeof(unsigned int), 1, file);

            //cout<<yxfolder.bytesPerVoxel<<endl;
        }
        fclose(file);
    }
    //
    return 0;
}

int BigTree::resume()
{
    // update zstart

    //
    struct stat info;

    if( stat( config4resume.c_str(), &info ) == 0 )
    {
        // config file exist
        ifstream infile;
        infile.open(config4resume.c_str());

        infile >> zstart;
        infile >> zpart;

        infile.close();
    }
    else
    {
        ofstream outfile;
        outfile.open(config4resume.c_str());

        outfile << 0 << endl;
        outfile << 1 << endl;

        outfile.close();
    }

    return 0;
}
