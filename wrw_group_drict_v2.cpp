#include <stdio.h>
#include <string.h>
#include <string.h>
#include "hip/hip_runtime.h"
#include "util.h"
#include "dev_util.h"

typedef long BB __attribute__((ext_vector_type(2)));


typedef  _Float16 HALF_T_;

// #define THREADS_NUM 1024
#define LDS_MAX_SIZE 15000
#define LDS_MIN_SIZE 5000

#define CVT_NUM_THREAD 8
#define MAX_FEATURE 256 * 256
#define REGISTER_SIZE  16
#define sub_block_single_block 16

typedef long BB __attribute__((ext_vector_type(2)));
typedef long longx2 __attribute__((ext_vector_type(2)));
typedef float floatx4 __attribute__((ext_vector_type(4)));


__attribute__((device)) static inline void bufferload_half_scatter(HALF_T_ *gA, 
                                                                    int *offsetA0,
                                                                    longx2 *globalReadA)
{
    asm volatile(
        "buffer_load_short_d16 %0,%1,%2,0, offen,offset:0\n"
        "s_waitcnt vmcnt(0)\n\t" 
        : "=v"(*gA), 
          "+v"(*offsetA0),
          "+s"(*globalReadA));
}

__attribute__((device)) static inline void
Dot2_F32_F16_1(int4 *A, int4 *B, float4 *C)
{
    asm volatile("v_dot2_f32_f16 %8,%0,%4,%8, \n"
                 "v_dot2_f32_f16 %9,%1,%5,%9, \n"
                 "v_dot2_f32_f16 %10,%2,%6,%10,\n"
                 "v_dot2_f32_f16 %11,%3,%7,%11,\n"
                 : "+v"(A->x),
                 "+v"(A->y),
                 "+v"(A->z),
                 "+v"(A->w),
                 "+v"(B->x), 
                 "+v"(B->y),
                 "+v"(B->z),
                 "+v"(B->w), 
                 "+v"(C->x),
                 "+v"(C->y),
                 "+v"(C->z),
                 "+v"(C->w));
}

__attribute__((device)) static inline void
Dot2_F32_F16_2(int4 *A,int4 *A1,int4 *B,int4 *B1,float4 *C,float4 *C1)
{
    asm volatile("v_dot2_f32_f16 %16,%0,%8,%16, \n"
                 "v_dot2_f32_f16 %17,%1,%9,%17, \n"
                 "v_dot2_f32_f16 %18,%2,%10,%18,\n"
                 "v_dot2_f32_f16 %19,%3,%11,%19,\n"
                 "v_dot2_f32_f16 %20,%4,%12,%20, \n"
                 "v_dot2_f32_f16 %21,%5,%13,%21, \n"
                 "v_dot2_f32_f16 %22,%6,%14,%22,\n"
                 "v_dot2_f32_f16 %23,%7,%15,%23,\n"
                 : "+v"(A->x),
                 "+v"(A->y),
                 "+v"(A->z),
                 "+v"(A->w),
                 "+v"(A1->x),
                 "+v"(A1->y),
                 "+v"(A1->z),
                 "+v"(A1->w),
                 "+v"(B->x), 
                 "+v"(B->y),
                 "+v"(B->z),
                 "+v"(B->w),
                 "+v"(B1->x), 
                 "+v"(B1->y),
                 "+v"(B1->z),
                 "+v"(B1->w),  
                 "+v"(C->x),
                 "+v"(C->y),
                 "+v"(C->z),
                 "+v"(C->w),
                 "+v"(C1->x),
                 "+v"(C1->y),
                 "+v"(C1->z),
                 "+v"(C1->w));
}
__attribute__((device)) static inline void bufferloadShort3(int4 *data,
                                                           int *offset0,
                                                           int *offset1,
                                                           int *offset2,
                                                           int *offset3,
                                                           int *offset4,
                                                           int *offset5,
                                                           int *offset6,
                                                           int *offset7,
                                                           BB *globalRead)
{
    float tmpVgpr0;
    float tmpVgpr1;
    float tmpVgpr2;
    float tmpVgpr3;

    asm volatile("buffer_load_short_d16 %0,%8,%16,0, offen offset:0\n"
                "buffer_load_short_d16 %1,%10,%16,0, offen offset:0\n"
                "buffer_load_short_d16 %2,%12,%16,0, offen offset:0\n"
                "buffer_load_short_d16 %3,%14,%16,0, offen offset:0\n"

                 "buffer_load_short_d16_hi %4,%9,%16,0, offen offset:0\n"
                 "buffer_load_short_d16_hi %5,%11,%16,0, offen offset:0\n"
                 "buffer_load_short_d16_hi %6,%13,%16,0, offen offset:0\n"
                 "buffer_load_short_d16_hi %7,%15,%16,0, offen offset:0\n"

                 "s_waitcnt vmcnt(0) \n"
                 "v_or_b32 %0, %0, %4 \n"
                 "v_or_b32 %1, %1, %5 \n"
                 "v_or_b32 %2, %2, %6 \n"
                 "v_or_b32 %3, %3, %7 \n"
                 : "+v"(data->x),
                   "+v"(data->y),
                   "+v"(data->z),
                   "+v"(data->w),
                   "+v"(tmpVgpr0),
                   "+v"(tmpVgpr1),
                   "+v"(tmpVgpr2),
                   "+v"(tmpVgpr3),
                   "+v"(*offset0),
                   "+v"(*offset1),
                   "+v"(*offset2),
                   "+v"(*offset3),
                   "+v"(*offset4),
                   "+v"(*offset5),
                   "+v"(*offset6),
                   "+v"(*offset7),
                   "+s"(*globalRead));               
}
__global__ void wrw_fp16_derict_group_v2(_Float16 *p_in,
                                        _Float16 *p_out,
                                        float *p_wei,
                                        KERNEL_DATA_INFO_T param)
                                            __attribute__((amdgpu_flat_work_group_size(1, 256)))
{
    int hi = param.DataFormat.IN_H;
    int wi = param.DataFormat.IN_W;
    int n  = param.DataFormat.NUMS;
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
    int thread_num  =  param.thread_num;

    // int group_single_block=param.group_single_block;
    // int bacths_Single_block=param.bacths_Single_block;
    // int subBlock_Single_Block=param.subBlock_Single_Block;

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


    BB globalReadA;
    globalReadA.x = (long)p_wei;
    globalReadA.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    BB globalReadB;
    globalReadB.x = (long)p_in;
    globalReadB.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));
    BB globalStoreC;
    globalStoreC.x = (long)p_out;
    globalStoreC.y = (((((long)0x20000) << 32) | 0xFFFFFFFE));

    BB globalReadA1 = globalReadA;
    BB globalReadB1 = globalReadB;

    int channels_single_group = channels / group;
    int kernels_single_group = kernelsNum / group;

    //假设有512个通道，256个卷积核，
    //分成64组
    // channelsBlockNum=64
    int channelsBlockNum = (channels + channels_single_group - 1) / channels_single_group;
    int batchNum = n;

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

    float result_value[REGISTER_SIZE] = {0.0f};
    int result_offset[REGISTER_SIZE];

    int feature_range = sub_in_height * sub_in_width;
    int filter_range = kernels_single_group * channels_single_group * fy * fx;
    int out_range = kernels_single_group * sub_height * sub_width;

    //------------------------------------------------------
    extern __shared__ _Float16 lds[];
    _Float16 *lds_out = lds;
    _Float16 *lds_feature = lds + out_range;

    int channnelsRange = channels_single_group;
    int kernelsRange = kernels_single_group;

    for (int tid = threadIdx.x; tid < out_range; tid += thread_num)
    {
        unsigned int k_index = tid / (sub_height * sub_width);
        unsigned int height_index = (tid % (sub_height * sub_width)) / sub_width;
        unsigned int width_index = tid % sub_width;
        //(oh,ow,k)
        lds_out[height_index * sub_width * kernelsRange + width_index * kernelsRange + k_index] = p_out[offset_p_out + k_index * ho * wo + height_index * wo + width_index];
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < feature_range; tid += thread_num)
    {
        unsigned int h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width + sub_in_height_start;
        unsigned int w_index = tid % sub_in_width + sub_in_width_start;
        unsigned int sub_h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width;
        unsigned int sub_w_index = tid % sub_in_width;
        int offset_flag = (h_index < hi && w_index < wi) ? 1 : 0;

        for (int c_index = 0; c_index < channnelsRange; c_index = c_index + 1)
        {
            HALF_T_ ggB1;

            int offset1 = (offset_flag == 1) ? 2 * (offset_p_in + c_index * hi * wi + h_index * wi + w_index) : -1;

            bufferload_half_scatter(&ggB1, &offset1, &globalReadB1);

            lds_feature[sub_h_index * sub_in_width * channnelsRange + sub_w_index * channnelsRange + c_index] = ggB1;
        }
    }

    __syncthreads();

    for (int tid = threadIdx.x; tid < filter_range; tid += thread_num)
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
                unsigned int offset_input = cur_h * (channnelsRange * sub_in_width) + (cur_w)*channnelsRange + c_index;
                // float in_item = (h_offset_flag == 1 && w_offset_flag == 1) ? lds_feature[offset_input] : 0;
                _Float16 in_item = lds_feature[offset_input];

                write_back_value += in_item * output_item;
            }
        }
        int indexOfWriteBackArray = tid / thread_num;
        //把偏移和中间值给保存下来
        result_offset[indexOfWriteBackArray] = (k_index + kernelIndex) * channels_single_group * fy * fx + (c_index)*fy * fx + r_index * fx + s_index;
        result_value[indexOfWriteBackArray] += write_back_value;

        // int writeback_offset = (k_index + kernelIndex) * channels_single_group * fy * fx + (c_index)*fy * fx + r_index * fx + s_index;
        // atomicAdd(p_wei + writeback_offset, write_back_value);
    }
    __syncthreads();

    for (int sub_block_offset = 1; sub_block_offset < sub_block_single_block; sub_block_offset++)
    {

        if (subBlockIndex + sub_block_offset > splitM * splitN - 1)
        {
            for (int tid = threadIdx.x; tid < filter_range; tid += thread_num)
            {

                int index = tid / thread_num;
                atomicAdd(p_wei + result_offset[index], result_value[index]);
            }
            return;
        }
        sub_block_row_idx = (subBlockIndex + sub_block_offset) / splitN;
        sub_block_col_idx = (subBlockIndex + sub_block_offset) % splitN;
        sub_block_base_height = ho / splitM;
        sub_block_base_width = wo / splitN;
        sub_block_height_res = ho % splitM;
        sub_block_width_res = wo % splitN;
        sub_block_base_rows =  splitM - sub_block_height_res;
        sub_block_base_cols =  splitN - sub_block_width_res;

        sub_height = sub_block_row_idx < sub_block_base_rows ? sub_block_base_height : sub_block_base_height  + 1;
        sub_width = sub_block_col_idx < sub_block_base_cols ? sub_block_base_width : sub_block_base_width  + 1;
        sub_height_start = sub_block_row_idx <= sub_block_base_rows ? sub_block_row_idx * sub_block_base_height :
                           sub_block_base_rows * sub_block_base_height + (sub_block_row_idx - sub_block_base_rows)*(sub_block_base_height + 1);                            
        sub_width_start = sub_block_col_idx <= sub_block_base_cols ? sub_block_col_idx * sub_block_base_width :
                           sub_block_base_cols * sub_block_base_width + (sub_block_col_idx - sub_block_base_cols)*(sub_block_base_width + 1);  


        sub_in_height_start = sy * sub_height_start + dy * 0 - py;
        sub_in_width_start = sx * sub_width_start + dx * 0 - px;
        sub_in_height_end = sy * (sub_height_start + sub_height - 1) + dy * fy - py;
        sub_in_width_end = sx * (sub_width_start + sub_width - 1) + dx * fx - px;

        sub_in_height = sub_in_height_end - sub_in_height_start;
        sub_in_width = sub_in_width_end - sub_in_width_start;

        offset_p_in = in * channels * hi * wi + channelsIndex * hi * wi;
        offset_p_out = in * kernelsNum * ho * wo + kernelIndex * ho * wo + sub_height_start * wo + sub_width_start;

        feature_range = sub_in_height * sub_in_width;
        out_range = kernels_single_group * sub_height * sub_width;

        lds_out = lds;
        lds_feature = lds + out_range;

        for (int tid = threadIdx.x; tid < out_range; tid += thread_num)
        {
            unsigned int k_index = tid / (sub_height * sub_width);
            unsigned int height_index = (tid % (sub_height * sub_width)) / sub_width;
            unsigned int width_index = tid % sub_width;
            lds_out[height_index * sub_width * kernelsRange + width_index * kernelsRange + k_index] = p_out[offset_p_out + k_index * ho * wo + height_index * wo + width_index];
        }
        __syncthreads();

        BB globalReadB2 = globalReadB;
        for (int tid = threadIdx.x; tid < feature_range; tid += thread_num)
        {
            unsigned int h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width + sub_in_height_start;
            unsigned int w_index = tid % sub_in_width + sub_in_width_start;
            unsigned int sub_h_index = (tid % (sub_in_width * sub_in_height)) / sub_in_width;
            unsigned int sub_w_index = tid % sub_in_width;
            int offset_flag = (h_index < hi && w_index < wi) ? 1 : 0;

            for (int c_index = 0; c_index < channnelsRange; c_index = c_index + 1)
            {
                HALF_T_ ggB1;

                int offset1 = (offset_flag == 1) ? 2 * (offset_p_in + c_index * hi * wi + h_index * wi + w_index) : -1;

                bufferload_half_scatter(&ggB1, &offset1, &globalReadB2);

                lds_feature[sub_h_index * sub_in_width * channnelsRange + sub_w_index * channnelsRange + c_index] = ggB1;
            }
        }

        __syncthreads();

        for (int tid = threadIdx.x; tid < filter_range; tid += thread_num)
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
                    unsigned int offset_input = cur_h * (channnelsRange * sub_in_width) + (cur_w)*channnelsRange + c_index;
                    // float in_item = (h_offset_flag == 1 && w_offset_flag == 1) ? lds_feature[offset_input] : 0;
                    _Float16 in_item = lds_feature[offset_input];

                    write_back_value += in_item * output_item;
                }
            }

            int indexOfWriteBackArray = tid / thread_num;
            //把偏移和中间值给保存下来
            result_offset[indexOfWriteBackArray] = (k_index + kernelIndex) * channels_single_group * fy * fx + (c_index)*fy * fx + r_index * fx + s_index;
            result_value[indexOfWriteBackArray] += write_back_value;
        }

        __syncthreads();
    }
    for (int tid = threadIdx.x; tid < filter_range; tid += thread_num)
    {

        int index = tid / thread_num;
        atomicAdd(p_wei + result_offset[index], result_value[index]);
    }

} // namespace hipnn

void run_wrw_fp16_derict_group_v2(_Float16 *d_Input_F16, _Float16 *d_OutputW_F16, float *d_FltW, const DATA_FORMAT_T *pDataFormat)
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

     int channels_single_group = channels / group;
    int kernels_single_group  = num_kernel / group;

    int group_single_block    = pDataFormat->group_single_block;
    int bacths_Single_block   = pDataFormat->bacths_Single_block;
    int subBlock_Single_Block = pDataFormat->subBlock_Single_Block;

    // int lds_size_approximate = (bacths_Single_block) * (channels_single_group) *
    //                            ((height + 2 * padh) * (width + 2 * padw) + o_h * o_w) *
    //                            sizeof(_Float16);
    int lds_size_approximate = (channels_single_group * (height + 2 * padh) * (width + 2 * padw) + kernels_single_group * o_h * o_w) * sizeof(_Float16);                           
    int count = 0;

    while(1)
    {
        int lds_size_tmp = lds_size_approximate / ((splitM) * (splitN));
        if(lds_size_tmp < LDS_MAX_SIZE)
        {
            break;
        }
        //支持
        // 1,1=1块
        // 2,1=2块
        // 2,2=4块
        // 4,2=8块
        // 4,4=16块
        // 8,4=32块
        // 8,8=64块
        // 16,8=128块
        // 16,16=256块
        if(count % 2 == 0&&splitM<16)
        {
            splitM = (splitM)*2;
        }
        if(count % 2 == 1&&splitN<16)
        {
            splitN = (splitN)*2;
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
    // int dynamic_lds_size = bacths_Single_block * group_single_block * (InSubBlockSize + max_outh * max_outw) * sizeof(_Float16);
    int dynamic_lds_size = (channels_single_group * InSubBlockSize + kernels_single_group * max_outh * max_outw) * sizeof(_Float16);


    int ChannelsBlockNum = (channels + channels_single_group - 1) / channels_single_group;
    int numgroupM        = ChannelsBlockNum * splitM * splitN;
    int numgroupN        = ((batch + bacths_Single_block - 1) / (bacths_Single_block));
    int numGroupZ        = 1;

    int filter_range = kernels_single_group * channels_single_group * r * s;

    int thread_num = 256;
    if(filter_range > 256 * REGISTER_SIZE)
    {
        thread_num = 512;
    }
    if(filter_range > 512 * REGISTER_SIZE)
    {
        thread_num = 1024;
    }

    dim3 GridDim(numgroupM, numgroupN, numGroupZ);
    dim3 BlockDim(thread_num, 1, 1);

    static int prt_flat = 1;
    if (prt_flat)
    {
        prt_flat = 0;
        printf(">>> run_wrw_fp16_derict_group_v2\r\n");
        printf("GridDim(%d,%d,%d) BlockDim(%d,%d,%d)\r\n", GridDim.x, GridDim.y, GridDim.z,
               BlockDim.x, BlockDim.y, BlockDim.z);
    }

    KERNEL_DATA_INFO_T KernelDataInfo;
    memset(&KernelDataInfo, 0, sizeof(KernelDataInfo));
    memcpy(&KernelDataInfo.DataFormat, pDataFormat, sizeof(KernelDataInfo.DataFormat));

    KernelDataInfo.splitM = splitM;
    KernelDataInfo.splitN = splitN;
    KernelDataInfo.thread_num = thread_num;
    KernelDataInfo.group_single_block  = group_single_block ;
    KernelDataInfo.bacths_Single_block  = bacths_Single_block ;
    KernelDataInfo.subBlock_Single_Block = subBlock_Single_Block;
    
    wrw_fp16_derict_group_v2<<<GridDim, BlockDim, dynamic_lds_size, 0>>>(d_Input_F16, d_OutputW_F16, d_FltW, KernelDataInfo);
}

void run_Transfer_float2float16_v2(_Float16 *dst, float *src, const DATA_FORMAT_T *pDataFormat)
{
    int k = pDataFormat->OUT_C;
    int c = pDataFormat->IN_C;
    int r = pDataFormat->KNL_H;
    int s = pDataFormat->KNL_W;
    int group = pDataFormat->GRPNUMS;

    int blockNum = ((c / group) * r * s * k - 1) / (256 * CVT_NUM_THREAD) + 1;
    int reminder = ((c / group) * r * s * k) % (256 * CVT_NUM_THREAD);

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

    float2float16<<<groupsCvt, threadsCvt, 0, 0>>>(dst, src, (c / group) * r * s * k, blockNum, reminder);
}