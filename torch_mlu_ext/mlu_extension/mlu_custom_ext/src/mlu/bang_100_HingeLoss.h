#pragma once
#include <cnrt.h>
void bang_100_HingeLoss_entry(float *predictions, float *targets,
                              float *output, int n);