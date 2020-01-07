#include "nn_functions.h"


/**
 * @brief Fast Q7 pointwise (1x1) convolution function
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in    input tensor dimention
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in,out]   bufferA      pointer to buffer space for input 
 *
 * @details
 * Changes from original function:
 * 1. Removed unused parameters.
 * 
 * Size of bufferA: 2 * ch_im_in
 * 
 * Constraints:
 *   Square input.
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 *
 */

void pointwise_conv_fast(const q7_t *Im_in,
                         const uint16_t dim_im_in,
                         const uint16_t ch_im_in,
                         const q7_t *wt,
                         const uint16_t ch_im_out,
                         const q7_t *bias,
                         const uint16_t bias_shift,
                         const uint16_t out_shift,
                         q7_t *Im_out,
                         q15_t *bufferA)
{

    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x;
    int16_t i_ch_out;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t *pBuffer = bufferA;
    q7_t *pOut = Im_out;

    for (i_out_y = 0; i_out_y < dim_im_in; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_in; i_out_x++)
        {
            /* This part implements the im2col function */
            arm_q7_to_q15_reordered_no_shift((q7_t *)Im_in + (i_out_y * dim_im_in + i_out_x) * ch_im_in, pBuffer,
                                             ch_im_in);
            pBuffer += ch_im_in;

            if (pBuffer == bufferA + 2 * ch_im_in)
            {
                pOut =
                    arm_nn_mat_mult_kernel_q7_q15_reordered(wt, bufferA, ch_im_out, ch_im_in, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
        {
            q31_t sum = bias[i_ch_out];
            q15_t *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t colCnt = ch_im_in >> 2;

            while (colCnt)
            {

                q31_t inA1, inA2;
                q31_t inB1, inB2;

                pA = (const q7_t *)read_and_pad_reordered((void *)pA, &inA1, &inA2);

                inB1 = *__SIMD32(pB)++;
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = *__SIMD32(pB)++;
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in & 0x3;
            while (colCnt)
            {
                q7_t inA1 = *pA++;
                q15_t inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut = (q7_t)__SSAT((sum >> out_shift), 8);
            pOut++;
        }
    }
}
