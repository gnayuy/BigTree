#ifndef __CUDALZWDECOMPKERNEL
#define __CUDALZWDECOMPKERNEL

#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <limits.h>

//
#define TABLESIZE 4095
#define LC 256
#define LE 257
#define DIMLEN 1024
#define PREFIXSUMMAX 512
#define INTERVALSIZE TABLESIZE-LE-2 //3836
#define TXSIZE INTERVALSIZE-1		//3835
#define PACKBIT 8
#define BIT09 253
#define BIT10 765
#define BIT11 1789
#define BIT12 3835
#define BITOFFSET09 253
#define BITOFFSET10 512
#define BITOFFSET11 1024


#define E_THREAD_NUM 32
#define ROWS_PER_STRIP 1
#define P_THREAD_NUM 128
#define PRETRANS 100000

#define PRETH 32
#define SUMTH (PRETH)>>1

//*********************************************************
//	GPU_PREDICTOR for grayscale or color(PlanarConfiguration=2)
//*********************************************************
__global__ void GPU_TIFFPredictor_gray(unsigned char *DECODE,unsigned int WIDTH,unsigned int LENGTH,int LANEBLOCKNUM)
{
    int laneId = threadIdx.x & 0x1f;
    int laneblockId=threadIdx.x>>5;
    int x=(2*blockIdx.x+laneblockId)*WIDTH*3+laneId;
    //int limit =(2*blockIdx.x+laneblockId+1)*WIDTH;
    int limit = (WIDTH+31)/32;
    unsigned int temp;
    unsigned int carry=0;
    unsigned int value ;
    //while(x < limit){
    if(LANEBLOCKNUM<=2*blockIdx.x+laneblockId) { return;}
    int lenmod3;
    if((LANEBLOCKNUM-1!=2*blockIdx.x+laneblockId) || ((lenmod3=LENGTH%3)==0)){
        for(int w=0;w<limit;w++){
            if(w*32+laneId<WIDTH)
                value = (DECODE[x]<<18)|(DECODE[x+WIDTH]<<9)|(DECODE[x+(WIDTH<<1)]);

            for (int i=1; i<=16; i=i<<1) {
                temp = __shfl_up(value, i, 32);
                if (laneId >= i)
                    value = (value + temp) & 0x3FDFEFF;
            }
            temp=(carry+value) & 0x3FDFEFF;
            if(w*32+laneId<WIDTH){
                DECODE[x]=(temp>>18);
                DECODE[x+WIDTH]=((temp>>9)&0xFF);
                DECODE[x+(WIDTH<<1)]=(temp&0xFF);
            }
            x=x+32;
            carry += __shfl(value, 31, 32);
            carry= (carry) & 0x3FDFEFF;
            //__syncthreads();
        }
    }else{
        for(int w=0;w<limit;w++){
            if(w*32+laneId<WIDTH){
                value = DECODE[x];
                for(int k=1;k<lenmod3;k++){
                    value = (value<<9)|DECODE[x+WIDTH*k];
                }
            }
            for (int i=1; i<=16; i=i<<1) {
                temp = __shfl_up(value, i, 32);
                if (laneId >= i)
                    value = (value + temp) & 0x3FDFEFF;
            }
            temp=(carry+value) & 0x3FDFEFF;
            if(w*32+laneId<WIDTH){
                for(int k=lenmod3-1;k>=0;k--){
                    DECODE[x+WIDTH*k]=(temp)&0xFF;
                    temp=temp>>9;
                }
            }
            x=x+32;
            carry += __shfl(value, 31, 32);
            carry= (carry) & 0x3FDFEFF;
            //__syncthreads();
        }
    }
}

//*********************************************************
//	GPU_LZW_DEC
//*********************************************************
__global__ void GPU_TiffLZWDecompression(
        unsigned char *CODE,
        unsigned char *DECODE,
        unsigned int STRIPSIZE,
        unsigned int StripOffsets_Offset,
        unsigned int StripByteCounts_Offset,
        unsigned int ByteOder)

{

    __shared__ unsigned int s_Preoffset[1024];     // used for Prefix-sum
    __shared__ short s_PCtabP[INTERVALSIZE];       // Pointer of Pointer-Character Table
    __shared__ short s_PCtabC[INTERVALSIZE];       // Character of Pointer-Character Table
    __shared__ short s_StrLength[4096];            // length of string in entry
    __shared__ short s_FirstChar[4096];            // first character of string in entry


    __shared__ int s_CCbefore4096flag;
    __shared__ unsigned int s_MaxBitOffset;        // max-bit-offset of strip
    __shared__ unsigned int s_CCIndex;             // ClearCode-flag (~=the number of Code in 1 segment)

    int currPointer;
    int prevPointer;


    unsigned int codeBitOffset;
    unsigned int codeByteOffset;
    int codeDigit;
    int mask[4]={0x1ff,0x3ff,0x7ff,0xfff};

    int bufStrLength;
    int bufFirstChar;

    int borderTraverse;
    int temp;

    int codeStrLength;
    int outputOffset;

    int codeIndex;
    unsigned int offsetSegment;
    unsigned int cudaBlockCount;
    unsigned int CCThreadIndex;

    int laneId=threadIdx.x&0x1f;
    int laneblockId=threadIdx.x>>5;

    if(threadIdx.x<256){
        s_FirstChar[threadIdx.x]=threadIdx.x;
        s_StrLength[threadIdx.x]=1;
    }

    if(threadIdx.x == 0){
        StripOffsets_Offset+=blockIdx.x*4;
        StripByteCounts_Offset+=blockIdx.x*4;
        if(ByteOder==77){ //ByteOrder == 'M'
            temp = (CODE[StripOffsets_Offset  ]<<24)|
                    (CODE[StripOffsets_Offset+1]<<16)|
                    (CODE[StripOffsets_Offset+2]<< 8)|
                    CODE[StripOffsets_Offset+3];
            s_MaxBitOffset = (CODE[StripByteCounts_Offset  ]<<24)|
                    (CODE[StripByteCounts_Offset+1]<<16)|
                    (CODE[StripByteCounts_Offset+2]<< 8)|
                    CODE[StripByteCounts_Offset+3];
        }else{
            temp = (CODE[StripOffsets_Offset+3]<<24)|
                    (CODE[StripOffsets_Offset+2]<<16)|
                    (CODE[StripOffsets_Offset+1]<< 8)|
                    CODE[StripOffsets_Offset  ];
            s_MaxBitOffset = (CODE[StripByteCounts_Offset+3]<<24)|
                    (CODE[StripByteCounts_Offset+2]<<16)|
                    (CODE[StripByteCounts_Offset+1]<< 8)|
                    CODE[StripByteCounts_Offset  ];
        }
        s_CCIndex = 9+9+PACKBIT*temp;  // stripoffset
        s_MaxBitOffset=(s_MaxBitOffset+temp-1)<<3;    // max-bit-offset of strip
    }
    __syncthreads();
    offsetSegment = s_CCIndex;               //Broadcasting first index of own strip
    outputOffset = blockIdx.x*STRIPSIZE;
    //DECODE[0]=s_MaxBitOffset+outputOffset;
    while(1){

        __syncthreads();
        if(threadIdx.x == 0 && offsetSegment-9<=s_MaxBitOffset){//threadaIdx.x
            codeByteOffset=(offsetSegment-9)>>3;
            prevPointer=0x1ff&(    ( ( CODE[codeByteOffset  ]<<(PACKBIT<<1) )|
                                     ( CODE[codeByteOffset+1]<< PACKBIT     )|
                                   ( CODE[codeByteOffset+2]               )  ) >> (PACKBIT*3-((offsetSegment-9)&0x7)-9));
            DECODE[outputOffset]=prevPointer;
            s_PCtabC[0]=prevPointer;
            s_FirstChar[258]=prevPointer;
            s_CCIndex=TXSIZE;
            s_CCbefore4096flag=0;
        }
        __syncthreads();
        cudaBlockCount=0;
        for(codeIndex=threadIdx.x;codeIndex<4096;codeIndex+=blockDim.x){
            if(codeIndex<BIT09){       //9bit Code
                codeDigit = 9;
                codeBitOffset=offsetSegment+codeIndex*9;
            }else if(codeIndex<BIT10){ //10bit Code
                codeDigit = 10;
                codeBitOffset=offsetSegment+BITOFFSET09*9+(codeIndex-BIT09)*10;
            }else if(codeIndex<BIT11){ //11bit Code
                codeDigit = 11;
                codeBitOffset=offsetSegment+BITOFFSET09*9+BITOFFSET10*10+(codeIndex-BIT10)*11;
            }else{				//12bit code
                codeDigit = 12;
                codeBitOffset=offsetSegment+BITOFFSET09*9+BITOFFSET10*10+BITOFFSET11*11+(codeIndex-BIT11)*12;
            }
            if(codeIndex<TXSIZE){
                if(s_MaxBitOffset<= codeBitOffset+codeDigit){
                    s_PCtabC[codeIndex+1]=-1;
                }else{
                    codeByteOffset= codeBitOffset>>3;
                    s_PCtabC[codeIndex+1]=mask[codeDigit-9]&(  (  (CODE[codeByteOffset  ]<<(PACKBIT<<1))|
                                                                  (CODE[codeByteOffset+1]<< PACKBIT    )|
                                                               (CODE[codeByteOffset+2]              )  )>>(PACKBIT*3-( codeBitOffset&0x7)-codeDigit));
                    s_FirstChar[258+codeIndex+1]=s_PCtabC[codeIndex+1];
                    if(s_PCtabC[codeIndex+1]==256){
                        s_CCbefore4096flag=1;
                        atomicMin(&s_CCIndex,codeIndex);
                    }
                }
            }
            __syncthreads();
            cudaBlockCount++;
            if(s_CCbefore4096flag==1)
                break;
            __syncthreads();
        }

        if(s_CCIndex<BIT09){
            offsetSegment+=s_CCIndex*9+9+9;
        }else if(s_CCIndex<BIT10){
            offsetSegment+=BITOFFSET09*9+(s_CCIndex-BIT09)*10+10+9;
        }else if(s_CCIndex<BIT11){
            offsetSegment+=BITOFFSET09*9+BITOFFSET10*10+(s_CCIndex-BIT10)*11+11+9;
        }else{
            offsetSegment+=BITOFFSET09*9+BITOFFSET10*10+BITOFFSET11*11+(s_CCIndex-BIT11)*12+12+9;
        }
        CCThreadIndex = (s_CCIndex-1)%(DIMLEN);
        codeIndex=threadIdx.x;
        borderTraverse=258;

        for(int count=1;count<=cudaBlockCount;count++){
            int returnflag=0;
            if(s_PCtabC[codeIndex+1]==-1&&s_CCIndex==TXSIZE)
                returnflag=1 ;
            if(s_CCIndex > codeIndex){
                currPointer=s_PCtabC[codeIndex+1];
                prevPointer=s_PCtabC[codeIndex];
                bufStrLength = 1;
                bufFirstChar = prevPointer;
            }else{
                currPointer=0;
                prevPointer=0;
                codeIndex=TXSIZE;
                bufFirstChar=0;
            }

            // ************************************************
            // table
            // ************************************************

            __syncthreads();
            while(borderTraverse <= bufFirstChar){
                bufFirstChar=s_FirstChar[bufFirstChar];
                bufStrLength++;
            }
            __syncthreads();
            s_StrLength[codeIndex+258]=bufStrLength+s_StrLength[bufFirstChar];
            s_FirstChar[codeIndex+258]=s_FirstChar[bufFirstChar];
            borderTraverse=borderTraverse+blockDim.x;
            __syncthreads();
            // ************************************************
            // prefix sum
            // ************************************************
            if(currPointer>=0){
                codeStrLength = s_StrLength[currPointer] ;
                s_PCtabC[codeIndex] = s_FirstChar[ currPointer ];
            }
            s_PCtabP[codeIndex] = prevPointer - 258;

            /*s_Preoffset[threadIdx.x] = codeStrLength ;
            temp = codeStrLength;
            __syncthreads();
            for(int i=1;i<DIMLEN;i=i<<1){
                if(i<=threadIdx.x)	//threadaIdx.x
                    temp += s_Preoffset[threadIdx.x - i];
                __syncthreads();
                s_Preoffset[threadIdx.x] = temp;
                __syncthreads();
            }*/
            int value=codeStrLength;
            for(int k=1;k<32;k=k<<1){
                int n = __shfl_up(value, k, 32);
                if (laneId >= k)
                    value += n;
            }
            if(laneId==31){
                s_Preoffset[laneblockId]=value;
            }
            __syncthreads();
            if(laneblockId==0){
                int value2=s_Preoffset[laneId];
                for(int k=1;k<32;k=k<<1){
                    int n = __shfl_up(value2, k, 32);
                    if (laneId >= k)
                        value2 += n;
                }
                s_Preoffset[laneId]=value2;
            }
            __syncthreads();
            if(laneblockId>0)
                value+=s_Preoffset[laneblockId-1];
            if(count==cudaBlockCount&&threadIdx.x==CCThreadIndex) s_Preoffset[31]=value;
            if(returnflag) return;
            // ************************************************
            // write<=PREFIXSUMMAX
            // ************************************************
            if(s_CCIndex > codeIndex){
                int tabIndex = currPointer - 258;
                value += outputOffset;
                for(int j=0 ; j<codeStrLength-1 ; j++){
                    DECODE[value-j] = s_PCtabC[tabIndex];
                    tabIndex = s_PCtabP[tabIndex];
                }
                DECODE[value-(codeStrLength-1)] = tabIndex + 258;
            }
            __syncthreads();
            if(count<cudaBlockCount)
                //outputOffset += s_Preoffset[blockDim.x-1];
                outputOffset += s_Preoffset[31];

            codeIndex += blockDim.x;

        }
        codeIndex = threadIdx.x;
        //outputOffset+=s_Preoffset[CCThreadIndex]+1;
        outputOffset+=s_Preoffset[31]+1;
    }

}

#endif
