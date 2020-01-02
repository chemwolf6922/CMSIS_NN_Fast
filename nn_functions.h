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

