#include "nn_functions.h"

/**
 * @brief Fast Q7 average pooling
 * @param[in]       Im_in       Pointer to the input tensor
 * @param[in]       dim_im_in   Input tensor dimention
 * @param[in]       ch_im_in    Input tensor channel
 * @param[in,out]   Im_out      Pointer to the output tensor
 * 
 * @details
 * Uses SIMD to calculate average pooling by 2*2 kernel.
 * 
 * Constrains:
 * 1. Square input
 * 2. Kernel size is 2.
*/

void avg_pool_q7_HWC_opt(q7_t* im_in,
                        const uint16_t dim_im_in,
                        const uint16_t ch_im_in,
                        q7_t* im_out)
{
    q7_t *pSrc1_1;
    q7_t *pSrc1_2;
    q7_t *pSrc2_1;
    q7_t *pSrc2_2;
    q7_t *pDst;
    
    pSrc1_1 = im_in;
    pSrc1_2 = im_in + ch_im_in;
    pSrc2_1 = im_in + ch_im_in*dim_im_in;
    pSrc2_2 = im_in + ch_im_in*dim_im_in + ch_im_in;
    pDst = im_out;

    uint32_t row_cnt = dim_im_in >> 1;
    while(row_cnt)
    {
        uint32_t col_cnt = dim_im_in >> 1;
        while(col_cnt)
        {
            uint32_t ch_cnt = ch_im_in >> 2;
            while(ch_cnt)
            {
                uint32_t in1,in2,out;
                in1 = *__SIMD32(pSrc1_1)++;
                in2 = *__SIMD32(pSrc1_2)++;
                out = __SHADD8(in1,in2);
                in1 = *__SIMD32(pSrc2_1)++;
                in2 = *__SIMD32(pSrc2_2)++;
                in1 = __SHADD8(in1,in2);
                out = __SHADD8(in1,out);
                *__SIMD32(pDst)++ = out;
                ch_cnt--;
            }
            pSrc1_1 += ch_im_in;
            pSrc1_2 += ch_im_in;
            pSrc2_1 += ch_im_in;
            pSrc2_2 += ch_im_in;
            col_cnt--;
        }
        pSrc1_1 += ch_im_in*dim_im_in;
        pSrc1_2 += ch_im_in*dim_im_in;
        pSrc2_1 += ch_im_in*dim_im_in;
        pSrc2_2 += ch_im_in*dim_im_in;
        row_cnt--;
    }
}
