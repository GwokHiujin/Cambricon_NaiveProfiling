#include <bang.h>
#include <cnrt.h>
#include "../common.h"


__mlu_global__ void AddKernel(float *output, const float *a,
                           const float *b, int32_t data_num);