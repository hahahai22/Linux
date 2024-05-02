/*
fp16对照fp32，保留fp16

*/

#include <stdio.h>
#include <string.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include "util.h"
#include "dev_util.h"

typedef  _Float16 HALF_T_;

typedef struct _HALFx4_T_
{
    HALF_T_ x;
    HALF_T_ y;
}_HALFx4_T_;

typedef struct _HALFx8__
{
    _HALFx4_T_ D0;
}_HALFx8_T_;


#define THREADS_NUM 256
#define LDS_MAX_SIZE 15000
#define LDS_MIN_SIZE 8000

#define MAX_FEATURE 256 * 256

typedef unsigned short half;


__attribute__((device)) static inline void bufferloadB(HALF_T_* gB, int* offsetB0, BB* globalReadB)
{
    asm volatile(

        // "buffer_load_short_d16 %0,%1,%2,0, idxen\n"
        "buffer_load_short_d16 %0,%1,%2,0, offen,offset:0\n"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(*gB),
          "+v"(*offsetB0),
          "+s"(*globalReadB));
}

// __attribute__((device)) static inline void bufferloadB1(HALF_T_* gB, int* offsetB0, int* offsetB1, BB* globalReadB)
// {
//     asm volatile(

//         "buffer_load_short_d16 %0,%2,%4,0, idxen\n"
//         "buffer_load_short_d16 %1,%3,%4,0, idxen\n"
//         "s_waitcnt vmcnt(0)\n\t" 
//         : "=v"(gB->D0.x),
//           "=v"(gB->D0.y),
//           "+v"(*offsetB0),
//           "+v"(*offsetB1),
//           "+s"(*globalReadB));
// }

__global__ void fp16_wrw_depthwise_5x5_v2(_Float16 *p_in,
                                                       _Float16 *p_out,
                                                       float *p_wei,
                                                       KERNEL_DATA_INFO_T param)
                                                       __attribute__((amdgpu_flat_work_group_size(1, 256)))
{

    /* param parse */
    // /********************** 1. 入参转换 ********************/
    // float* p_wei   =  (float* )FltW;
    // _Float16* p_in  = (_Float16*)Input;
    // _Float16* p_out = (_Float16*)OutputW;
    

    int hi = param.DataFormat.IN_H;
    int wi = param.DataFormat.IN_W;
    // int n  = param.DataFormat.NUMS;
    int ho = param.DataFormat.OUT_H;
    int wo = param.DataFormat.OUT_W;
    int sy = param.DataFormat.STRD_H;
    int sx = param.DataFormat.STRD_W;
    int dy = param.DataFormat.DIA_H;
    int dx = param.DataFormat.DIA_W;
    int py = param.DataFormat.PAD_H;
    int px = param.DataFormat.PAD_W;
    int fy = param.DataFormat.KNL_H;
    int fx = param.DataFormat.KNL_W;
    int group = param.DataFormat.GRPNUMS;   
    int channels =  param.DataFormat.IN_C; 
    int kernelsNum = param.DataFormat.OUT_C;

    int splitM     =  param.splitM;
    int splitN     =  param.splitN;
    int group_single_block = param.DataFormat.group_single_block;
    int bacths_Single_block = param.DataFormat.bacths_Single_block;
    int sub_block_single_block = param.DataFormat.subBlock_Single_Block;
    //printf("******!*!*!*!*!*!*!*!**!!* group_single_block = %d, bacths_Single_block = %d, subBlock_Single_Block = %d\n",group_single_block,bacths_Single_block, subBlock_Single_Block);

    int blockSplitNum = (splitM * splitN + sub_block_single_block - 1) / sub_block_single_block;
    int subBlockIndex = (blockIdx.x % blockSplitNum) * sub_block_single_block; //子块的索引


    int sub_block_row_idx = subBlockIndex / splitN;
    int sub_block_col_idx = subBlockIndex % splitN;
    int sub_block_base_height = ho / splitM;
    int sub_block_base_width = wo / splitN;
    int sub_block_height_res = ho % splitM;
    int sub_block_width_res = wo % splitN;
    int sub_block_base_rows =  splitM - sub_block_height_res;
    int sub_block_base_cols =  splitN - sub_block_width_res;

    int sub_height = sub_block_row_idx < sub_block_base_rows ? sub_block_base_height : sub_block_base_height  + 1;
    int sub_width = sub_block_col_idx < sub_block_base_cols ? sub_block_base_width : sub_block_base_width  + 1;
    int sub_height_start = sub_block_row_idx <= sub_block_base_rows ? sub_block_row_idx * sub_block_base_height :
                           sub_block_base_rows * sub_block_base_height + (sub_block_row_idx - sub_block_base_rows)*(sub_block_base_height + 1);                            
    int sub_width_start = sub_block_col_idx <= sub_block_base_cols ? sub_block_col_idx * sub_block_base_width :
                           sub_block_base_cols * sub_block_base_width + (sub_block_col_idx - sub_block_base_cols)*(sub_block_base_width + 1);  

    // longx2 globalReadA, globalReadB;
    // globalReadA.x = (long)p_out;
    // globalReadA.x = (globalReadA.x | (((long)(0x2 << 16)) << 32)); 
    // globalReadA.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    // globalReadB.x = (long)p_in;
    // globalReadB.x = (globalReadB.x | (((long)(0x2 << 16)) << 32)); 
    // globalReadB.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    longx2 globalReadA;
    globalReadA.x = (long)p_wei;
    globalReadA.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    longx2 globalReadB;
    globalReadB.x = (long)p_in;
    globalReadB.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    longx2 globalStoreC;
    globalStoreC.x = (long)p_out;
    globalStoreC.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));

    int channels_single_group = channels / group;
    int kernels_single_group = kernelsNum / group;

    //假设有512个通道，256个卷积核，
    //分成64组
    // channelsBlockNum=64
    int channelsBlockNum = (channels + channels_single_group - 1) / channels_single_group;

    int in = blockIdx.y;                                                       // n的索引
    int channelsIndex = (blockIdx.x / blockSplitNum) * channels_single_group;  // channels的索引

    int groupIndex = channelsIndex / channels_single_group; // groups的索引
    int kernelIndex = groupIndex * kernels_single_group;


    int sub_in_height_start = sy * sub_height_start + dy * 0 - py;
    int sub_in_width_start = sx * sub_width_start + dx * 0 - px;
    int sub_in_height_end =
        sy * (sub_height_start + sub_height - 1) + dy * fy - py; //这个end是取不到的，是开区间
    int sub_in_width_end =
        sx * (sub_width_start + sub_width - 1) + dx * fx - px; //这个end是取不到的，是开区间

    int sub_in_height = sub_in_height_end - sub_in_height_start;
    int sub_in_width = sub_in_width_end - sub_in_width_start;

    //先做不分块的吧
    //那么in矩阵有两个偏移,一个n的偏移，一个c的偏移（由group决定）,如果不分块的话，那么就没有h和w的偏移
    // out矩阵里有两个偏移，一个n的偏移，一个k的偏移（由group决定），如果不分块的话，就没有oh和ow的偏移
    // weight矩阵中有一个偏移，就是k的偏移,也由group决定
    int offset_p_in = in * channels * hi * wi + channelsIndex * hi * wi;
    int offset_p_out = in * kernelsNum * ho * wo + kernelIndex * ho * wo + sub_height_start * wo + sub_width_start;
    int offset_p_weight = kernelIndex * channels * fy * fx;

    float result_value[16];
    int result_offset[16];
    result_value[0] = 0.0;
    result_value[1] = 0.0;
    result_value[2] = 0.0;
    result_value[3] = 0.0;
    result_value[4] = 0.0;
    result_value[5] = 0.0;
    result_value[6] = 0.0;
    result_value[7] = 0.0;
    result_value[8] = 0.0;
    result_value[9] = 0.0;
    result_value[10] = 0.0;
    result_value[11] = 0.0;
    result_value[12] = 0.0;
    result_value[13] = 0.0;
    result_value[14] = 0.0;
    result_value[15] = 0.0;


    int feature_range = sub_in_height * sub_in_width;
    int filter_range = kernels_single_group * channels_single_group * fy * fx;
    int out_range = kernels_single_group * sub_height * sub_width;

    //------------------------------------------------------
    extern __shared__ _Float16 lds[];
    _Float16 *lds_out = lds;
    _Float16 *lds_feature = lds + out_range;

    int channelsRange = channels_single_group;
    int kernelsRange = kernels_single_group;

    for (int tid = threadIdx.x; tid < out_range; tid += THREADS_NUM)
    {
        unsigned int k_index = tid / (sub_height * sub_width);
        unsigned int height_index = (tid % (sub_height * sub_width)) / sub_width;
        unsigned int width_index = tid % sub_width;
        //(oh,ow,k)
        lds_out[height_index * sub_width * kernelsRange + width_index * kernelsRange + k_index] = p_out[offset_p_out + k_index * ho * wo + height_index * wo + width_index];
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < feature_range; tid += THREADS_NUM)
    {
        unsigned int h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width + sub_in_height_start;
        unsigned int w_index = tid % sub_in_width + sub_in_width_start;
        unsigned int sub_h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width;
        unsigned int sub_w_index = tid % sub_in_width;
        int offset_flag = (h_index < hi && w_index < wi) ? 1 : 0;

        for (int c_index = 0; c_index < channelsRange; c_index = c_index + 1)
        {
            HALF_T_ ggB1;

            int offset1 = (offset_flag == 1) ? 2 * (offset_p_in + c_index * hi * wi + h_index * wi + w_index) : -1;
            bufferloadB(&ggB1, &offset1, &globalReadB);

            lds_feature[sub_h_index * sub_in_width * channelsRange + sub_w_index * channelsRange + c_index] = ggB1;
        }
    }

    __syncthreads();

    for (int tid = threadIdx.x; tid < filter_range; tid += THREADS_NUM)
    {

        //把循环次数拆成k,c,r,s,每个线程算kcrs其中的一个点
        //假设是8×4×3×3，那么就是288个点
        int r_index = tid / (fx * channels_single_group * kernels_single_group);
        int s_index = (tid % (fx * channels_single_group * kernels_single_group)) / (kernels_single_group * channels_single_group);
        int k_index = (tid % (channels_single_group * kernels_single_group)) / (channels_single_group);
        int c_index = tid % channels_single_group;

        //        int height_start_index = sub_in_height_start + dy * r_index + py;
        //        int width_start_index = sub_in_width_start + dx * s_index + px;
        int height_start_index = dy * r_index;
        int width_start_index = dx * s_index;
        float write_back_value = 0.0;

        for (int iheight = 0; iheight < sub_height; iheight++)
        {
            for (int iwidth = 0; iwidth < sub_width; iwidth++)
            {

                // out矩阵在lds中的偏移
                int offset_output = k_index + iheight * sub_width * kernelsRange + iwidth * kernelsRange;
                _Float16 output_item = lds_out[offset_output];

                // in矩阵在lds中的偏移
                // int h_offset_flag = (height_start_index + sy * iheight >= 0 && height_start_index + sy * iheight < hi) ? 1 : 0;
                // int w_offset_flag = (width_start_index + sx * iwidth >= 0 && width_start_index + sx * iwidth < wi) ? 1 : 0;

                unsigned int cur_h = height_start_index + sy * iheight;
                unsigned int cur_w = width_start_index + sx * iwidth;
                //定义unsigned，如果有负数，会有一个很大的值，很大的值一定大于原来的范围
                unsigned int offset_input = cur_h * (channelsRange * sub_in_width) + (cur_w)*channelsRange + c_index;
                // float in_item = (h_offset_flag == 1 && w_offset_flag == 1) ? lds_feature[offset_input] : 0;
                _Float16 in_item = lds_feature[offset_input];

                write_back_value += in_item * output_item;
            }
        }
        int indexOfWriteBackArray = tid / THREADS_NUM;
        //把偏移和中间值给保存下来
        result_offset[indexOfWriteBackArray] = (k_index + kernelIndex) * channels_single_group * fy * fx + (c_index)*fy * fx + r_index * fx + s_index;
        result_value[indexOfWriteBackArray] += write_back_value;

        // int writeback_offset = (k_index + kernelIndex) * channels_single_group * fy * fx + (c_index)*fy * fx + r_index * fx + s_index;
        // atomicAdd(p_wei + writeback_offset, write_back_value);
    }
    __syncthreads();
   
    for (int tid = threadIdx.x; tid < filter_range; tid += THREADS_NUM)
    {

        int index = tid / THREADS_NUM;
        atomicAdd(p_wei + result_offset[index], result_value[index]);
    }
}

void run_fp16_wrw_depthwise_5x5_v2(_Float16 *d_Input_F16, _Float16 *d_OutputW_F16, float *d_FltW, const DATA_FORMAT_T *pDataFormat)
{
    int batch = pDataFormat->NUMS;
    int channels = pDataFormat->IN_C;
    int height = pDataFormat->IN_H;
    int width = pDataFormat->IN_W;
    int num_kernel = pDataFormat->OUT_C;
    int r = pDataFormat->KNL_H;
    int s = pDataFormat->KNL_W;
    int padh = pDataFormat->PAD_H;
    int padw = pDataFormat->PAD_W;
    int stride_h  = pDataFormat->STRD_H;
    int stride_w  = pDataFormat->STRD_W;
    int dilate_h = pDataFormat->DIA_H;
    int dilate_w = pDataFormat->DIA_W;
    int group = pDataFormat->GRPNUMS;

    int splitM     = 1;
    int splitN     = 1;

    int dilate_filter_h = dilate_h * (r - 1) + 1;
    int dilate_filter_w = dilate_w * (s - 1) + 1;

    int o_h = (height + 2 * padh - dilate_filter_h) / stride_h + 1;
    int o_w = (width + 2 * padw - dilate_filter_w) / stride_w + 1;

    //  int channels_single_group = channels / group;
    // int kernels_single_group  = num_kernel / group;

    //每个块计算多少组
    int group_single_block = pDataFormat->group_single_block;
    //每个块计算多少batch
    int bacths_Single_block = pDataFormat->bacths_Single_block;
    //每个块里面有多少子块
    int subBlock_Single_Block = pDataFormat->subBlock_Single_Block;

    int channels_single_group = channels / group;
    int kernels_single_group  = num_kernel / group;

    int lds_size_approximate = bacths_Single_block * group_single_block *
                               (channels_single_group * (height + 2 * padh) * (width + 2 * padw) + kernels_single_group * o_h * o_w) *
                               sizeof(_Float16);
    int count = 0;

    int lds_size_tmp = lds_size_approximate / ((splitM) * (splitN));
    while(lds_size_tmp > LDS_MAX_SIZE)
    {
        lds_size_tmp = lds_size_approximate / ((splitM) * (splitN));

        if(count % 2 == 0)
        {
            splitM = (splitM) * 2;
        }
        if(count % 2 == 1)
        {
            splitN = (splitN) * 2;
        }
        count++;
    }

    Sub_Block sub_obj[splitM * splitN];
    int ret_max_index    = SplitBlock(o_h, o_w, splitM, splitN, sub_obj);
    int InSubBlockSize   = GetInSubBlockSize(stride_h,
                                           stride_w,
                                           padh,
                                           padw,
                                           dilate_h,
                                           dilate_w,
                                           r,
                                           s,
                                           sub_obj[ret_max_index].sub_height,
                                           sub_obj[ret_max_index].sub_width);

    int max_outh         = sub_obj[ret_max_index].sub_height;
    int max_outw         = sub_obj[ret_max_index].sub_width;
    int dynamic_lds_size = bacths_Single_block * (channels_single_group * InSubBlockSize + kernels_single_group * max_outh * max_outw) * sizeof(_Float16);

    int ChannelsBlockNum = (channels + channels_single_group - 1) / channels_single_group;
    int numgroupM        = ChannelsBlockNum * splitM * splitN;
    int numgroupN        = ((batch + bacths_Single_block - 1) / (bacths_Single_block));
    int numGroupZ        = 1;
    dim3 GridDim(numgroupM, numgroupN, numGroupZ);
    dim3 BlockDim(THREADS_NUM, 1, 1);

    static int prt_flat = 1;
    if (prt_flat)
    {
        prt_flat = 0;
        printf(">>> run_fp16_wrw_depthwise_5x5_v2\r\n");
        printf("GridDim(%d,%d,%d) BlockDim(%d,%d,%d)\r\n", GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z);
    }

    KERNEL_DATA_INFO_T KernelDataInfo;
    memset(&KernelDataInfo, 0, sizeof(KernelDataInfo));
    memcpy(&KernelDataInfo.DataFormat, pDataFormat, sizeof(KernelDataInfo.DataFormat));

    printf("splitM: %d, splitN: %d\n", splitM, splitN);

    KernelDataInfo.splitM = splitM;
    KernelDataInfo.splitN = splitN;
    KernelDataInfo.DataFormat.group_single_block = group_single_block ;
    KernelDataInfo.DataFormat.bacths_Single_block = bacths_Single_block ;
    KernelDataInfo.DataFormat.subBlock_Single_Block = subBlock_Single_Block;
    
    fp16_wrw_depthwise_5x5_v2<<<GridDim, BlockDim, dynamic_lds_size, 0>>>(d_Input_F16, d_OutputW_F16, d_FltW, KernelDataInfo);
}

void run_Transfer_float2float16_v2(_Float16 *dst, float *src, const DATA_FORMAT_T *pDataFormat)
{
    int k = pDataFormat->OUT_C;
    int c = pDataFormat->IN_C;
    int r = pDataFormat->KNL_H;
    int s = pDataFormat->KNL_W;
    int g = pDataFormat->GRPNUMS;

    int blockNum = (c / g * r * s * k - 1) / (256 * CVT_NUM_THREAD) + 1;
    int reminder = (c / g * r * s * k) % (256 * CVT_NUM_THREAD);

    dim3 threadsCvt(256, 1, 1);
    dim3 groupsCvt(blockNum, 1, 1);

    static int prt_flat = 1;
    if (prt_flat)
    {
        prt_flat = 0;
        printf(">>> run_Transfer_float2float16_v2\r\n");
        printf("GridDim(%d,%d,%d) BlockDim(%d,%d,%d) reminder: %d\r\n", groupsCvt.x, groupsCvt.y, groupsCvt.z,
               threadsCvt.x, threadsCvt.y, threadsCvt.z, reminder);
    }

    float2float16<<<groupsCvt, threadsCvt, 0, 0>>>(dst, src, c / g * r * s * k, blockNum, reminder);
}
