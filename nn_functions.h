#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

void pointwise_conv_basic(const q7_t *Im_in,
                          const uint16_t dim_im_in,
                          const uint16_t ch_im_in,
                          const q7_t *wt,
                          const uint16_t ch_im_out,
                          const q7_t *bias,
                          const uint16_t bias_shift,
                          const uint16_t out_shift,
                          q7_t *Im_out,
                          q15_t *bufferA);

void pointwise_conv_fast(const q7_t *Im_in,
                         const uint16_t dim_im_in,
                         const uint16_t ch_im_in,
                         const q7_t *wt,
                         const uint16_t ch_im_out,
                         const q7_t *bias,
                         const uint16_t bias_shift,
                         const uint16_t out_shift,
                         q7_t *Im_out,
                         q15_t *bufferA);


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
              q15_t *bufferB);

void depthwise_conv(const q7_t *Im_in,
                    const uint16_t dim_im_in,
                    const uint16_t ch_im_in,
                    const q7_t *wt,
                    const uint16_t dim_kernel,
                    const uint16_t padding,
                    const uint16_t out_shift,
                    q7_t *Im_out,
                    q7_t *bufferA);

void avg_pool_q7_HWC_opt(q7_t* im_in,
                        const uint16_t dim_im_in,
                        const uint16_t ch_im_in,
                        q7_t* im_out);


