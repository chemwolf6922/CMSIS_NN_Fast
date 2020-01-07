#include "nn_functions.h"

/**
 * @brief Fast depthwise convolution
 * @param[in]       Im_in       Pointer to the input tensor
 * @param[in]       dim_im_in   Input tensor dimention
 * @param[in]       ch_im_in    Input tensor channel
 * @param[in]       wt          Pointer to kernel weights
 * @param[in]       ch_im_out   Output tensor channel
 * @param[in]       dim_kernel  Kernel dimention
 * @param[in]       padding     'Same' padding only, please caluclate that
 * @param[in]       bias        Pointers to bias
 * @param[in]       bias_shift  
 * @param[in]       out_shift
 * @param[in,out]   Im_out      Pointer to the output tensor
 * @param[in,out]   bufferA     Pointer to buffer A (tensor buffer)    
 * @param[in,out]   bufferB     Pointer to buffer B (weight buffer)
 * 
 * @details
 * Changes from the original function:
 * 1. Optimized memory copy process with a trade off of larger bufferA.
 * 
 * BufferA size:  dim_kernel * (dim_kernel - 1 + dim_im_in) * ch_im_in (in q15)
 * BufferB size:  dim_kernel * dim_kernel * ch_im_in * ch_im_out (in q15)
 * 
 * Constrains:
 * 1. Square input
 * 2. Output channel is even
 * 3. ch_im_in is even
*/

void conv_HWC(const q7_t *Im_in,
              const uint16_t dim_im_in,
              const uint16_t ch_im_in,
              const q7_t *wt,
              const uint16_t ch_im_out,
              const uint16_t dim_kernel,
              const uint16_t padding,
              const q7_t *bias,
              const uint16_t bias_shift,
              const uint16_t out_shift,
              q7_t *Im_out,
              q15_t *bufferA,
              q15_t *bufferB)
{

    // Move parameters to bufferB
    int32_t x, y;
    uint32_t data_to_transfer;
    q15_t *pBuffer;
    q7_t *data_source;
    pBuffer = bufferB;
    data_source = (q7_t *)wt;
    data_to_transfer = dim_kernel * dim_kernel * ch_im_in * ch_im_out;
    arm_q7_to_q15_no_shift(data_source, pBuffer, data_to_transfer);

    // Move data and calculate per col
    pBuffer = bufferA;
    uint32_t num_data_in_row = dim_kernel * ch_im_in;
    uint32_t num_data_in_im_row = dim_im_in * ch_im_in;
    int32_t para_per_ch_out = dim_kernel * dim_kernel * ch_im_in;

    //set bottom and top padding
    memset((void *)pBuffer, 0, num_data_in_row * padding * 2); // *2 for q15
    memset((void *)(pBuffer + num_data_in_row * (padding + dim_im_in)), 0, num_data_in_row * padding * 2);

    for (x = 0; x < dim_im_in; x++)
    {
        //Move data to bufferA
        pBuffer = bufferA;
        if (x < padding)
        {
            //set the left padding
            memset((void *)(pBuffer + num_data_in_row * padding), 0, num_data_in_row * dim_im_in * 2);
            data_to_transfer = ch_im_in * (dim_kernel - padding + x);
            pBuffer += num_data_in_row * padding + num_data_in_row - data_to_transfer;
            data_source = (q7_t *)Im_in;
            for (y = 0; y < dim_im_in; y++)
            {
                arm_q7_to_q15_no_shift(data_source, pBuffer, data_to_transfer);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }
        else if (x > (dim_im_in - padding - 1))
        {
            //set the right padding
            memset((void *)(pBuffer + num_data_in_row * padding), 0, num_data_in_row * dim_im_in * 2);
            data_to_transfer = ch_im_in * (dim_kernel - (x + padding - (dim_im_in - 1)));
            pBuffer += num_data_in_row * padding;
            data_source = (q7_t *)Im_in + (x - padding) * ch_im_in;
            for (y = 0; y < dim_im_in; y++)
            {
                arm_q7_to_q15_no_shift(data_source, pBuffer, data_to_transfer);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }
        else
        {
            pBuffer += num_data_in_row * padding;
            data_source = (q7_t *)Im_in + (x - padding) * ch_im_in;
            for (y = 0; y < dim_im_in; y++)
            {
                arm_q7_to_q15_no_shift(data_source, pBuffer, num_data_in_row);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }

        //Calculation
        //Calculate two points at the same time
        for (y = 0; y < dim_im_in; y += 2)
        {
            q7_t *pOut = Im_out + x * ch_im_out + ch_im_out * dim_im_in * y;
            q7_t *pOut2 = pOut + ch_im_out * dim_im_in;
            q7_t *pBias = (q7_t*)bias;

            uint16_t chCnt = ch_im_out >> 1;
            q15_t *pPara = bufferB;
            q15_t *pPara2 = bufferB + para_per_ch_out;
            //Calculate two channels at the same time
            while (chCnt > 0)
            {
                q15_t *pData = bufferA + num_data_in_row * y;
                q15_t *pData2 = pData + num_data_in_row;

                q31_t sum1 = (q31_t)((*pBias) << bias_shift);
                q31_t sum2 = sum1;
                q31_t sum3 = (q31_t)((*pBias+1) << bias_shift);
                q31_t sum4 = sum3;

                int32_t paraCnt = para_per_ch_out >> 2;
                while (paraCnt)
                {
                    q31_t inB1 = *__SIMD32(pData)++;
                    q31_t inB2 = *__SIMD32(pData2)++;

                    q31_t inA1 = *__SIMD32(pPara)++;
                    q31_t inA2 = *__SIMD32(pPara2)++;

                    sum1 = __SMLAD(inA1, inB1, sum1);
                    sum2 = __SMLAD(inA1, inB2, sum2);
                    sum3 = __SMLAD(inA2, inB1, sum3);
                    sum4 = __SMLAD(inA2, inB2, sum4);

                    inB1 = *__SIMD32(pData)++;
                    inB2 = *__SIMD32(pData2)++;

                    inA1 = *__SIMD32(pPara)++;
                    inA2 = *__SIMD32(pPara2)++;

                    sum1 = __SMLAD(inA1, inB1, sum1);
                    sum2 = __SMLAD(inA1, inB2, sum2);
                    sum3 = __SMLAD(inA2, inB1, sum3);
                    sum4 = __SMLAD(inA2, inB2, sum4);

                    paraCnt--;
                }
                paraCnt = para_per_ch_out & 0x3U;
                while (paraCnt)
                {
                    q15_t inA1 = *pPara++;
                    q15_t inB1 = *pData++;
                    q15_t inA2 = *pPara2++;
                    q15_t inB2 = *pData2++;

                    sum1 += inA1 * inB1;
                    sum2 += inA1 * inB2;
                    sum3 += inA2 * inB1;
                    sum4 += inA2 * inB2;
                    paraCnt--;
                }

                *pOut = (q7_t)__SSAT((sum1 >> out_shift), 8);
                *(pOut + 1) = (q7_t)__SSAT((sum3 >> out_shift), 8);
                *pOut2 = (q7_t)__SSAT((sum2 >> out_shift), 8);
                *(pOut2 + 1) = (q7_t)__SSAT((sum4 >> out_shift), 8);
                pBias += 2;
                pOut += 2;
                pOut2 += 2;
                pPara += para_per_ch_out;
                pPara2 += para_per_ch_out;
                chCnt--;
            }
        }
    }
}

