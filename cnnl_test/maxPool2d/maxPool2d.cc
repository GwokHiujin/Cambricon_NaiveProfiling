#include <stdio.h>
#include <time.h>
#include <iostream>
#include "../public/tool.h"

// Reflect to CNNLPoolingForwardWithIndex
static cnnlStatus_t TestPooling(cnnlHandle_t handle,
                                Optensor &optensor,
                                cnnlPoolingMode_t PoolingMode,
                                cnnlDataType_t dtype)
{
    cnrtQueue_t queue = NULL;
    CNNL_CHECK(cnnlGetQueue(handle, &queue));
    cnrtNotifier_t start, end; // Record mlu time
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));

    int dim_test[4] = {16, 256, 112, 112};
    int dim_transpose[4] = {16, 112, 112, 256};
    int dim_kernel[2] = {3, 3};
    int stride[2] = {2, 2};
    int padding[4] = {0, 0, 0, 0};
    int dilation[2] = {1, 1};
    int dim_result[4];
    dim_result[0] = dim_test[0];
    dim_result[1] = floor((dim_test[2] + padding[0] + padding[1] - dim_kernel[0]) / stride[0]) + 1;
    dim_result[2] = floor((dim_test[3] + padding[2] + padding[3] - dim_kernel[1]) / stride[1]) + 1;
    dim_result[3] = dim_test[1]; 

    cnnlTensorDescriptor_t descInput;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descInput));
    CNNL_CHECK(cnnlSetTensorDescriptor(descInput, CNNL_LAYOUT_NCHW, dtype, 4, dim_test));

    cnnlTensorDescriptor_t descInput_;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descInput_));
    CNNL_CHECK(cnnlSetTensorDescriptor(descInput_, CNNL_LAYOUT_NHWC, dtype, 4, dim_transpose));

    cnnlTensorDescriptor_t descResult;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descResult));
    CNNL_CHECK(cnnlSetTensorDescriptor(descResult, CNNL_LAYOUT_NHWC, dtype, 4, dim_result));

    cnnlTensorDescriptor_t descIndex;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descIndex));
    CNNL_CHECK(cnnlSetTensorDescriptor(descIndex, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT32, 4, dim_result));

    const int permute[4] = {0, 2, 3, 1};   // Transpose NCHW to NHWC
    cnnlTransposeDescriptor_t descTranspose;
    CNNL_CHECK(cnnlCreateTransposeDescriptor(&descTranspose));
    CNNL_CHECK(cnnlSetTransposeDescriptor(descTranspose, 4, permute));

    cnnlPoolingDescriptor_t descPooling;
    CNNL_CHECK(cnnlCreatePoolingDescriptor(&descPooling));
    CNNL_CHECK(cnnlSetPoolingNdDescriptor_v2(descPooling, PoolingMode, CNNL_NOT_PROPAGATE_NAN, 4, dim_kernel, padding, stride, dilation, false));

    float_t *device_input_ptr = NULL;
    float_t *device_input_ptr_ = NULL;
    float_t *device_result_ptr = NULL;
    int32_t *device_index_ptr = NULL;
    size_t dims_test = 1;
    for (size_t i = 0; i < 4; i++)
    {
        dims_test *= dim_test[i];
    }
    size_t size_test = dims_test * sizeof(float_t *);
    size_t dims_result = 1;
    for (size_t i = 0; i < 4; i++)
    {
        dims_result *= dim_result[i];
    }
    size_t size_result = dims_result * sizeof(float_t *);
    CNRT_CHECK(cnrtMalloc((void **)&device_input_ptr, size_test));
    CNRT_CHECK(cnrtMalloc((void **)&device_input_ptr_, size_test));
    CNRT_CHECK(cnrtMalloc((void **)&device_index_ptr, size_result));
    CNRT_CHECK(cnrtMalloc((void **)&device_result_ptr, size_result));

    float_t *host_input = (float_t *)malloc(size_test);
    float_t *host_input_ = (float_t *)malloc(size_test);
    int32_t *host_index = (int32_t *)malloc(size_result);
    int32_t *MLU_index = (int32_t *)malloc(size_result);
    float_t *host_result = (float_t *)malloc(size_result);
    float_t *MLU_result = (float_t *)malloc(size_result);

    for (size_t i = 0; i < dims_test; i++)
    {
        host_input[i] = RAND_MAX / rand();
    }

    HostTimer copyin_timer;
    copyin_timer.start();
    CNRT_CHECK(cnrtMemcpy(device_input_ptr, host_input, size_test, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(device_input_ptr_, host_input_, size_test, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(device_index_ptr, host_index, size_result, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(device_result_ptr, host_result, size_result, CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin_timer.stop();
    optensor.memcpyH2D_time = copyin_timer.tv_usec;

    // MLU compute
    HostTimer host_timer;
    size_t size_workspace = 0;
    CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize(handle, descInput, descResult, &size_workspace));
    void *workspace = NULL;
    if (size_workspace > 0)
    {
        CNRT_CHECK(cnrtMalloc((void **)&workspace, size_workspace));
    }
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));
    host_timer.start();
    for (size_t i = 0; i < 100; i++)
    {
        CNNL_CHECK(cnnlTranspose(handle, descTranspose, descInput, device_input_ptr, descInput_, device_input_ptr_));
        CNNL_CHECK(cnnlPoolingForwardWithIndex(handle, descPooling,
                                               NULL, descInput_, device_input_ptr_,
                                               NULL, descResult, device_result_ptr,
                                               descIndex, device_index_ptr,
                                               workspace, size_workspace));
    }
    host_timer.stop();
    optensor.host_time = host_timer.tv_usec;
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue)); // Remember to sync queue! Or the time record may be inaccurate.
    CNRT_CHECK(cnrtNotifierDuration(start, end, &(optensor.mlu_compute_time)));

    HostTimer copyout_timer;
    copyout_timer.start();
    CNRT_CHECK(cnrtMemcpy(MLU_result, device_result_ptr, size_result, CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(MLU_index, device_index_ptr, size_result, CNRT_MEM_TRANS_DIR_DEV2HOST));
    copyout_timer.stop();
    optensor.memcpyD2H_time = copyout_timer.tv_usec;

    // CNRT Free
    CNRT_CHECK(cnrtFree(device_index_ptr));
    CNRT_CHECK(cnrtFree(device_result_ptr));
    CNRT_CHECK(cnrtFree(device_input_ptr));
    CNRT_CHECK(cnrtFree(device_input_ptr_));
    if (workspace)
    {
        CNRT_CHECK(cnrtFree(workspace));
    }
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));

    // CNNL Free
    CNNL_CHECK(cnnlDestroyPoolingDescriptor(descPooling));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descInput));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descInput_));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descIndex));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descResult));
    free(host_input);
    free(host_input_);
    free(host_index);
    free(host_result);
    free(MLU_result);

    return CNNL_STATUS_SUCCESS;
}

int main(int argc, char *argv[])
{
    Optensor optensor;
    cnnlPoolingMode_t PoolingMode = CNNL_POOLING_MAX;
    cnnlDataType_t dtype = CNNL_DTYPE_FLOAT;

    // Init device
    int dev;
    CNRT_CHECK(cnrtGetDevice(&dev));
    CNRT_CHECK(cnrtSetDevice(dev));

    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));

    // Create CNNL Handle
    cnnlHandle_t handle;
    CNNL_CHECK(cnnlCreate(&handle));
    CNNL_CHECK(cnnlSetQueue(handle, queue));

    // Test
    CNNL_CHECK(TestPooling(handle, optensor, PoolingMode, dtype));

    CNRT_CHECK(cnrtQueueDestroy(queue));
    CNNL_CHECK(cnnlDestroy(handle));

    // Print result
    std::stringstream host_time;
    host_time << "Host Time(us): " << optensor.host_time / 100;
    std::stringstream mlu_compute_time;
    mlu_compute_time << "MLU Time(ms): " << optensor.mlu_compute_time / 100000;
    std::stringstream memcpy_time;
    memcpy_time << "CopyIn Time(us): " << optensor.memcpyH2D_time << "; CopyOut Time(us): " << optensor.memcpyD2H_time;

    LOG(host_time.str());
    LOG(mlu_compute_time.str());
    LOG(memcpy_time.str());

    return 0;
}
