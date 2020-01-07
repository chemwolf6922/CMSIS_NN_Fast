#include "nn_functions.h"

/**
 * @brief Fast depthwise convolution
 * @param[in]       Im_in       Pointer to the input tensor
 * @param[in]       dim_im_in   Input tensor dimention
 * @param[in]       ch_im_in    Input tensor channel
 * @param[in]       wt          Pointer to kernel weights
 * @param[in]       dim_kernel  Kernel dimention
 * @param[in]       padding     'Same' padding only, please caluclate that
 * @param[in,out]   Im_out      Pointer to the output tensor
 * @param[in,out]   bufferA     Pointer to buffer A     
 * 
 * @details
 * Changes from the original function:
 * 1. Optimized memory copy process with a trade off of larger bufferA.
 * 2. Remove bias parameters, depthwise layer (tensorflow) does not have bias.
 * 
 * BufferA size:  dim_kernel * (dim_kernel - 1 + dim_im_in) * ch_im_in
 * 
 * Constrains:
 * 1. Square input
*/

void depthwise_conv(const q7_t *Im_in,
                    const uint16_t dim_im_in,
                    const uint16_t ch_im_in,
                    const q7_t *wt,
                    const uint16_t dim_kernel,
                    const uint16_t padding,
                    const uint16_t out_shift,
                    q7_t *Im_out,
                    q7_t *bufferA)
{

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x;
    q7_t *pBuffer = bufferA;
    q7_t *pOut;
    q7_t *data_source;
    uint16_t rowCnt;
    uint16_t row_shift;

    uint16_t num_data_in_row = dim_kernel * ch_im_in;
    uint16_t data_to_transfer;
    uint32_t num_data_in_im_row = dim_im_in * ch_im_in;
    uint16_t colCnt;
    q7_t *pB;
    q7_t *pA;
    //TODO: ignore the mul with zero to accelerate
    //set the top and bottom padding
    memset((void *)pBuffer, 0, num_data_in_row * padding);
    memset((void *)(pBuffer + num_data_in_row * (padding + dim_im_in)), 0, num_data_in_row * padding);

    for (i_out_x = 0; i_out_x < dim_im_in; i_out_x++)
    {
        pBuffer = bufferA;
        if (i_out_x < padding)
        {
            //set the left padding
            memset((void *)(pBuffer + num_data_in_row * padding), 0, num_data_in_row * dim_im_in);
            data_to_transfer = ch_im_in * (dim_kernel - padding + i_out_x);
            pBuffer += num_data_in_row * padding + num_data_in_row - data_to_transfer;
            data_source = (q7_t *)Im_in;
            for (i_out_y = 0; i_out_y < dim_im_in; i_out_y++)
            {
                memcpy(pBuffer, data_source, data_to_transfer);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }
        else if (i_out_x > (dim_im_in - padding - 1))
        {
            //set the right padding
            memset((void *)(pBuffer + num_data_in_row * padding), 0, num_data_in_row * dim_im_in);
            data_to_transfer = ch_im_in * (dim_kernel - (i_out_x + padding - (dim_im_in - 1)));
            pBuffer += num_data_in_row * padding;
            data_source = (q7_t *)Im_in + (i_out_x - padding) * ch_im_in;
            for (i_out_y = 0; i_out_y < dim_im_in; i_out_y++)
            {
                memcpy(pBuffer, data_source, data_to_transfer);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }
        else
        {
            pBuffer += num_data_in_row * padding;
            data_source = (q7_t *)Im_in + (i_out_x - padding) * ch_im_in;
            for (i_out_y = 0; i_out_y < dim_im_in; i_out_y++)
            {
                memcpy(pBuffer, data_source, num_data_in_row);
                data_source += num_data_in_im_row;
                pBuffer += num_data_in_row;
            }
        }

        for (i_out_y = 0; i_out_y < dim_im_in; i_out_y++)
        {
            rowCnt = ch_im_in >> 2;
            row_shift = 0;
            pOut = Im_out + i_out_x * ch_im_in + i_out_y * num_data_in_im_row;
            while (rowCnt)
            {
                q31_t sum = 0;
                q31_t sum2 = 0;
                q31_t sum3 = 0;
                q31_t sum4 = 0;

                colCnt = (dim_kernel * dim_kernel) >> 1;
                pB = bufferA + num_data_in_row * i_out_y + row_shift;
                pA = (q7_t *)wt + row_shift;

                row_shift += 4;

                while (colCnt)
                {
                    q31_t inA1, inA2, inB1, inB2, opA, opB;

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHTB(opB, inB1, 16);
                    inB1 = __PKHBT(inB1, opB, 16);
                    inA1 = *__SIMD32(pA);
                    pA += ch_im_in;
                    opB = *__SIMD32(pA);
                    pA += ch_im_in;
                    inA2 = __PKHTB(opB, inA1, 16);
                    inA1 = __PKHBT(inA1, opB, 16);
                    opA = __SXTB16(inA1);
                    opB = __SXTB16(inB1);
                    sum = __SMLAD(opA, opB, sum);
                    opA = __SXTB16(__ROR(inA1, 8));
                    opB = __SXTB16(__ROR(inB1, 8));
                    sum2 = __SMLAD(opA, opB, sum2);
                    opA = __SXTB16(inA2);
                    opB = __SXTB16(inB2);
                    sum3 = __SMLAD(opA, opB, sum3);
                    opA = __SXTB16(__ROR(inA2, 8));
                    opB = __SXTB16(__ROR(inB2, 8));
                    sum4 = __SMLAD(opA, opB, sum4);
                    colCnt--;
                }

                colCnt = (dim_kernel * dim_kernel) & 0x1;
                while (colCnt)
                {
                    union arm_nnword inA, inB;
                    inA.word = *__SIMD32(pA);
                    pA += ch_im_in;
                    inB.word = *__SIMD32(pB);
                    pB += ch_im_in;
                    sum += inA.bytes[0] * inB.bytes[0];
                    sum2 += inA.bytes[1] * inB.bytes[1];
                    sum3 += inA.bytes[2] * inB.bytes[2];
                    sum4 += inA.bytes[3] * inB.bytes[3];
                    colCnt--;
                }

                *pOut++ = (q7_t)__SSAT((sum  >> out_shift), 8);
                *pOut++ = (q7_t)__SSAT((sum2 >> out_shift), 8);
                *pOut++ = (q7_t)__SSAT((sum3 >> out_shift), 8);
                *pOut++ = (q7_t)__SSAT((sum4 >> out_shift), 8);

                rowCnt--;
            }

            rowCnt = ch_im_in & 0x3;
            while (rowCnt)
            {
                pB = bufferA + num_data_in_row * i_out_y + row_shift;
                pA = (q7_t *)wt + row_shift;
                q31_t sum = 0;
                uint16_t colCnt = (dim_kernel * dim_kernel);

                row_shift += 1;

                while (colCnt)
                {
                    q7_t A1 = *pA;
                    q7_t B1 = *pB;
                    pA += ch_im_in;
                    pB += ch_im_in;
                    sum += A1 * B1;

                    colCnt--;
                }
                *pOut++ = (q7_t)__SSAT((sum >> out_shift), 8);
                rowCnt--;
            }
        }
    }
}
