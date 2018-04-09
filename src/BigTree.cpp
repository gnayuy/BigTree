// BigTree.cpp

#include "BigTree.h"

//
BLOCK::BLOCK()
{
    lengthFileName = 25;
    lengthDirName = 21;

    bWrite = false;
}

BLOCK::~BLOCK()
{
    fileNames.clear();
    offsets_D.clear();
}

int BLOCK::findNonZeroBlocks()
{
    if(nonZeroBlocks.size()<1)
    {
        cout<<"No block found \n";
        return -1;
    }

    //
    for(int i=0; i<nonZeroBlocks.size(); i++)
    {
        if(nonZeroBlocks[i]==true)
        {
            bWrite = true;
            break;
        }
    }

    //
    return 0;
}

LAYER::LAYER()
{

}

LAYER::~LAYER()
{
    blocks.clear();
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
BigTree::BigTree(string inputdir, string outputdir, int scales)
{
    // default parameters settings
    block_width = 256;
    block_height = 256;
    block_depth = 256;

    nbits = 4;

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
        cout<<"mkdir "<<dstdir<<endl;

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
                            ((uint32)(depth/pow(2,halve_pow2[res_i])))  / n_stacks_D[res_i] + (stack_sli < ((int)(depth/pow(2,halve_pow2[res_i])))  % n_stacks_D[res_i] ? 1:0);
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
    z_max_res = max(min(MAX_IMAGES_STREAM,(int)block_depth/2),(int)pow(2,halve_pow2[resolutions-1]));
    if ( (z_max_res > 1) && z_max_res > block_depth/2 )
    {
        cout<<"too much resolutions "<<resolutions<<endl;
        return -1;
    }
    z_ratio=depth/z_max_res;

    //
    return 0;
}

uint8 *BigTree::load(long zs, long ze)
{
    //
    long sbv_V, sbv_H, sbv_D;

    sbv_V = height;
    sbv_H = width;
    sbv_D = ze - zs;
    //D0 = zs;

    //
    uint8 *subvol = NULL;

    try
    {
        subvol = new uint8 [sbv_V * sbv_H* sbv_D * datatype];
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
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel
    {
        #pragma omp for
        for(k=0; k<sbv_D; k++)
        {
            unsigned int sx, sy;
            readTiff(dataInMemory[k],imgList[k],sx,sy,0,0,0,sbv_V-1,0,sbv_H-1);
        }
    }

    //
    return subvol;
}

int BigTree::reformat()
{
    //
    int stack_block[TMITREE_MAX_HEIGHT];
    int slice_start[TMITREE_MAX_HEIGHT];
    int slice_end[TMITREE_MAX_HEIGHT];

    for(int res_i=0; res_i< resolutions; res_i++)
    {
        stack_block[res_i] = 0;
        slice_start[res_i] = 0;
        slice_end[res_i] = slice_start[res_i] + stacks_D[res_i][0][0][0] - 1;
    }

    //
    for(long z=0, z_parts=1; z<depth; z+=z_max_res, z_parts++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ubuffer = load(z,(z+z_max_res <= depth) ? (z+z_max_res) : depth);

        auto end = std::chrono::high_resolution_clock::now();

        cout<<"load a sub volume takes "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<" ms."<<endl;

        //
        if(nbits)
        {
            long tot_size = (height * width * ((z_parts<=z_ratio) ? z_max_res : (depth%z_max_res)))*color;
            if ( datatype == 2 )
            {
                #pragma omp parallel
                {
                    uint16 *ptr = (uint16 *) ubuffer;
                    #pragma omp for
                    for ( long i=0; i<tot_size; i++ ) {
                        ptr[i] = ptr[i] >> nbits << 1;
                    }
                }
            }
        }

        //saving the sub volume
        start = std::chrono::high_resolution_clock::now();
        for(int i=0; i< resolutions; i++)
        {
            // meta info for index()
            LAYER layer;
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

            //compute the number of slice of previous groups at resolution i
            //note that z_parts in the number and not an index (starts from 1)
            long n_slices_pred  = (z_parts - 1) * z_max_res / pow(2,halve_pow2[i]);

            //buffer size along D is different when the remainder of the subdivision by z_max_res is considered
            long z_size = (z_parts<=z_ratio) ? z_max_res : (depth%z_max_res);

            //halvesampling resolution if current resolution is not the deepest one
            if(i!=0)
            {
                if ( halve_pow2[i] == (halve_pow2[i-1]+1) )
                {
                    halveSample(ubuffer,(int)height/(pow(2,i-1)),(int)width/(pow(2,i-1)),(int)z_size/(pow(2,halve_pow2[i-1])),HALVE_BY_MAX,datatype);
                }
                else if ( halve_pow2[i] == halve_pow2[i-1] )
                {
                    halveSample(ubuffer,(int)height/(pow(2,i-1)),(int)width/(pow(2,i-1)),(int)z_size/(pow(2,halve_pow2[i-1])),HALVE_BY_MAX,datatype);
                }
                else
                {
                    cout<<"halve sampling level "<<halve_pow2[i]<<" not supported at resolution "<<i<<endl;
                    return -1;
                }
            }

            //saving at current resolution if it has been selected and iff buffer is at least 1 voxel (Z) deep
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

                    int sz[4];
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

                                //
                                int slice_start_temp = 0;
                                for ( int j=0; j < n_stacks_D[i]; j++ ) {
                                    sz[2] = stacks_D[i][stack_row][stack_column][j];

                                    std::stringstream abs_pos_z_temp;
                                    abs_pos_z_temp.width(6);
                                    abs_pos_z_temp.fill('0');
                                    abs_pos_z_temp << (int)((pow(2,halve_pow2[i])*slice_start_temp) * 10);

                                    std::stringstream img_path_temp;
                                    img_path_temp << H_DIR_path.str() << "/" << multires_merging_x_pos.str() << "_" << multires_merging_y_pos.str() << "_" << abs_pos_z_temp.str()<<".tif";

                                    //cout<<"z "<<z<<" ("<<sz[0]<<", "<<sz[1]<<", "<<sz[2]<<") "<<abs_pos_z_temp.str()<<endl;

                                    // auto start_init = std::chrono::high_resolution_clock::now();
                                    if(nCopies==0)
                                    {
                                        if( initTiff3DFile((char *)img_path_temp.str().c_str(),sz[0],sz[1],sz[2],sz[3],datatype_out) != 0)
                                        {
                                            cout<<"fail in initTiff3DFile\n";
                                            return -1;
                                        }
                                        srcFile = img_path_temp.str();
                                    }
                                    else
                                    {
                                        copyFile(srcFile.c_str(), img_path_temp.str().c_str());
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

                        // meta info for index()
                        BLOCK block;

                        block.width = sz[0];
                        block.height = sz[1];
                        block.depth = sz[2];
                        block.color = sz[3];
                        block.bytesPerVoxel = datatype_out;

                        block.dirName = multires_merging_x_pos.str() + "/" + multires_merging_x_pos.str() + "_" + multires_merging_y_pos.str();
                        block.offset_H = start_width;
                        block.offset_V = start_height;
                        block.fileNames.push_back(multires_merging_x_pos.str() + "_" + multires_merging_y_pos.str() + "_" + abs_pos_z.str() + ".tif");
                        block.offsets_D.push_back(z);

                        bool blocksaved = false;

                        //cout<<"z "<<z<<endl;

                        // WARNING: assumes that block size along z is not less that z_size/(powInt(2,i))
                        for(int buffer_z=0; buffer_z<z_size/(pow(2,halve_pow2[i])); buffer_z++, slice_ind++)
                        {
                            // D0 must be subtracted because z is an absolute index in volume while slice index should be computed on a relative basis (i.e. starting form 0)
                            if ( (z / pow(2,halve_pow2[i]) + buffer_z) > slice_end[i] && !block_changed)
                            {
                                // start a new block along z
                                std::stringstream abs_pos_z_next;
                                abs_pos_z_next.width(6);
                                abs_pos_z_next.fill('0');
                                abs_pos_z_next << (pow(2,halve_pow2[i])*(slice_end[i]+1)) * 10;
                                img_path.str("");
                                img_path << partial_img_path.str() << abs_pos_z_next.str() << ".tif";

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
                            if ( datatype == 2 )
                            {
                                // 16-bit input

                                if(datatype_out == 1)
                                {
                                    // 8-bit output

                                    //
                                    uint16 *raw_ch16 = (uint16 *) ubuffer;

                                    //
                                    #pragma omp parallel for collapse(2)
                                    for(long i=0; i<sz[1]; i++)
                                    {
                                        //uint8* row_data_8bit = p + i*sz[0];

                                        for(long j=0; j<sz[0]; j++)
                                        {
                                            p[i*sz[0]+j] = raw_ch16[(i+start_height)*(raw_img_width) + (j+start_width)];
                                        }
                                    }

                                    // check all-zeros array
                                    int numNonZeros = 0;

                                    #pragma omp parallel for
                                    for(int i=0; i<szChunk; i++)
                                    {
                                        numNonZeros |= p[i];
                                    }

                                    if(numNonZeros != 0)
                                    {
                                        int temp_n_chans = color;
                                        if(temp_n_chans==2)
                                            temp_n_chans++;

                                        appendSlice2Tiff3DFile(fhandle,slice_ind,(unsigned char *)p,sz[0],sz[1],temp_n_chans,8,sz[2]);
                                        blocksaved = true;
                                    }
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
                        block.nonZeroBlocks.push_back(blocksaved);
                        block.nBlocksPerDir = block.fileNames.size();
                        layer.blocks.push_back(block);
                    }
                    start_height += stacks_V[i][stack_row][0][0];
                }
            }

            //
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

    //
    for(int res_i=0; res_i< resolutions; res_i++)
    {
        //
        string filename = filePaths[res_i] + "/mdata.bin";

        //
        LAYER layer = meta.layers[res_i];


//        //
//        ofstream outfile(filename.c_str(), ios::out | ios::app | ios::binary);

//        if(outfile.is_open())
//        {
//            outfile << meta.mdata_version;
//            outfile << meta.reference_V;
//            outfile << meta.reference_H;
//            outfile << meta.reference_D;

//            outfile << layer.vs_x;
//            outfile << layer.vs_y;
//            outfile << layer.vs_z;
//            outfile << layer.vs_x;
//            outfile << layer.vs_y;
//            outfile << layer.vs_z;

//            outfile << meta.org_V;
//            outfile << meta.org_H;
//            outfile << meta.org_D;

//            outfile << layer.dim_V;
//            outfile << layer.dim_H;
//            outfile << layer.dim_D;

//            outfile << layer.rows;
//            outfile << layer.cols;

//            int n = layer.blocks.size(); // rows * cols

//            cout<<"test "<<n<<" = "<<layer.rows*layer.cols<<endl;

//            for(int i=0; i<n; i++)
//            {
//                BLOCK block = layer.blocks[i];
//                uint32 N_BLOCKS = block.nBlocksPerDir;

//                outfile << block.height;
//                outfile << block.width;
//                outfile << block.depth;

//                outfile << N_BLOCKS;

//                outfile << block.color;
//                outfile << block.offset_V;
//                outfile << block.offset_H;
//                outfile << block.lengthDirName;
//                outfile << block.dirName;

//                for(int j=0; j<N_BLOCKS; j++)
//                {
//                    outfile << block.lengthFileName;
//                    outfile << block.fileNames[j];
//                    outfile << block.depth;
//                    outfile << block.offsets_D[j];
//                }
//                outfile << block.bytesPerVoxel;
//            }

//            //
//            outfile.close();
//        }
//        else
//        {
//            cout<<"fail in write file "<<filename<<endl;
//            return -1;
//        }

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

        int n = layer.blocks.size(); // rows * cols

        for(int i=0; i<n; i++)
        {
            BLOCK block = layer.blocks[i];
            uint32 N_BLOCKS = block.nBlocksPerDir;

            if(block.findNonZeroBlocks())
            {
                continue;
            }

            fwrite(&(block.height), sizeof(uint32), 1, file);
            fwrite(&(block.width), sizeof(uint32), 1, file);
            fwrite(&(block.depth), sizeof(uint32), 1, file);
            fwrite(&N_BLOCKS, sizeof(uint32), 1, file);
            fwrite(&(block.color), sizeof(uint32), 1, file);
            fwrite(&(block.offset_V), sizeof(int), 1, file);
            fwrite(&(block.offset_H), sizeof(int), 1, file);
            fwrite(&(block.lengthDirName), sizeof(uint16), 1, file);
            fwrite(const_cast<char *>(block.dirName.c_str()), block.lengthDirName, 1, file);

            for(int j=0; j<N_BLOCKS; j++)
            {
                if(block.nonZeroBlocks[j]==false)
                {
                    if( remove( block.fileNames[j].c_str() ) != 0 )
                    {
                        cout<<"Error deleting file \n";
                        return -1;
                    }
                }
                else
                {
                    fwrite(&(block.lengthFileName), sizeof(uint16), 1, file);
                    fwrite(const_cast<char *>(block.fileNames[j].c_str()), block.lengthFileName, 1, file);
                    fwrite(&(block.depth), sizeof(uint32), 1, file);
                    fwrite(&(block.offsets_D[j]), sizeof(int), 1, file);
                }
            }
            fwrite(&(block.bytesPerVoxel), sizeof(uint32), 1, file);
        }
        fclose(file);
    }
    //
    return 0;
}

