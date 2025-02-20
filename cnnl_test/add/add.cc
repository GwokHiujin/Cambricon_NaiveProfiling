#include <stdio.h>
#include <time.h>
#include <iostream>
#include "../public/tool.h"

// Reflect to CNNLOpTensor
template <typename T>
static cnnlStatus_t TestOpTensor(cnnlHandle_t handle,
                                 Optensor &optensor,
                                 cnnlOpTensorDesc_t OpTensorDesc,
                                 cnnlDataType_t dtype,
                                 T alpha1,
                                 T alpha2,
                                 T beta)
{
    cnrtQueue_t queue = NULL;
    CNNL_CHECK(cnnlGetQueue(handle, &queue));
    cnrtNotifier_t start, end; // Record mlu time
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));

    int dim_test[4] = {16, 256, 112, 112}; // NCHW Type
    cnnlTensorDescriptor_t descX;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descX));
    CNNL_CHECK(cnnlSetTensorDescriptor(descX, CNNL_LAYOUT_NCHW, dtype, 4, dim_test));

    cnnlTensorDescriptor_t descY;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descY));
    CNNL_CHECK(cnnlSetTensorDescriptor(descY, CNNL_LAYOUT_NCHW, dtype, 4, dim_test));

    cnnlTensorDescriptor_t descZ;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&descZ));
    CNNL_CHECK(cnnlSetTensorDescriptor(descZ, CNNL_LAYOUT_NCHW, dtype, 4, dim_test));

    cnnlOpTensorDescriptor_t descOpTensor; // Prepare desc of OpTensor
    CNNL_CHECK(cnnlCreateOpTensorDescriptor(&descOpTensor));
    CNNL_CHECK(cnnlSetOpTensorDescriptor(descOpTensor, OpTensorDesc, dtype, CNNL_NOT_PROPAGATE_NAN));

    bool isNull = false;
    T *device_X_ptr = NULL;
    T *device_Y_ptr = NULL;
    T *device_Z_ptr = NULL;
    size_t dims_test = 16 * 256 * 112 * 112;
    size_t size_test = dims_test * sizeof(T *); // data_size (count * sizeof[dtype])
    CNRT_CHECK(cnrtMalloc((void **)&device_X_ptr, size_test));
    CNRT_CHECK(cnrtMalloc((void **)&device_Y_ptr, size_test));
    CNRT_CHECK(cnrtMalloc((void **)&device_Z_ptr, size_test));

    T *host_X = (T *)malloc(size_test);
    T *host_Y = (T *)malloc(size_test);
    T *host_result = (T *)malloc(size_test);
    T *MLU_result = (T *)malloc(size_test);

    for (size_t i = 0; i < dims_test; i++)
    {
        // Prepare input data
        host_X[i] = rand() / RAND_MAX;
        host_Y[i] = rand() / RAND_MAX;
        host_result[i] = host_X[i] + host_Y[i]; // Save for accuracy check
    }

    HostTimer copyin_timer; // Actually, we can use cnperf to record the detailed time, but it produces tmp database.
    copyin_timer.start();
    CNRT_CHECK(cnrtMemcpy(device_X_ptr, host_X, size_test, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(device_Y_ptr, host_Y, size_test, CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(device_Z_ptr, host_result, size_test, CNRT_MEM_TRANS_DIR_HOST2DEV));
    copyin_timer.stop();
    optensor.memcpyH2D_time = copyin_timer.tv_usec;

    HostTimer host_timer;
    size_t size_workspace = 0;
    CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(handle, descX, descY, descZ, &size_workspace));
    void *workspace = NULL;
    if (size_workspace > 0)
    {
        CNRT_CHECK(cnrtMalloc((void **)&workspace, size_workspace));
    }
    CNRT_CHECK(cnrtPlaceNotifier(start, queue));
    host_timer.start();
    CNNL_CHECK(cnnlOpTensor(handle, descOpTensor,
                            &alpha1, descX, device_X_ptr,
                            &alpha2, descY, device_Y_ptr,
                            workspace, size_workspace,
                            &beta, descZ, device_Z_ptr));
    host_timer.stop();
    optensor.host_time = host_timer.tv_usec;
    CNRT_CHECK(cnrtPlaceNotifier(end, queue));
    CNRT_CHECK(cnrtQueueSync(queue)); // Remember to sync queue! Or the time record may be inaccurate.
    CNRT_CHECK(cnrtNotifierDuration(start, end, &(optensor.mlu_compute_time)));

    HostTimer copyout_timer;
    copyout_timer.start();
    CNRT_CHECK(cnrtMemcpy(MLU_result, device_Z_ptr, size_test, CNRT_MEM_TRANS_DIR_DEV2HOST));
    copyout_timer.stop();
    optensor.memcpyD2H_time = copyout_timer.tv_usec;

    // Accuracy check
    optensor.isPass = MSE(host_result, MLU_result, size_test, isNull);

    // CNRT Free
    CNRT_CHECK(cnrtFree(device_X_ptr));
    CNRT_CHECK(cnrtFree(device_Y_ptr));
    CNRT_CHECK(cnrtFree(device_Z_ptr));
    if (workspace)
    {
        CNRT_CHECK(cnrtFree(workspace));
    }
    CNRT_CHECK(cnrtNotifierDestroy(start));
    CNRT_CHECK(cnrtNotifierDestroy(end));

    // CNNL Free
    CNNL_CHECK(cnnlDestroyOpTensorDescriptor(descOpTensor));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descX));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descY));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(descZ));
    free(host_X);
    free(host_Y);
    free(host_result);
    free(MLU_result);
    return CNNL_STATUS_SUCCESS;
}

int main(int argc, char *argv[])
{
    Optensor optensor;
    cnnlOpTensorDesc_t OpTensorDesc = CNNL_OP_TENSOR_ADD; // 0: CNNL_OP_TENSOR_ADD
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
    CNNL_CHECK(TestOpTensor(handle, optensor, OpTensorDesc, dtype, std::stof("1"), std::stof("1"), std::stof("1")));

    CNRT_CHECK(cnrtQueueDestroy(queue));
    CNNL_CHECK(cnnlDestroy(handle));

    // Print result
    if (optensor.isPass == 1)
    {
        LOG("PASSED");
        std::stringstream host_time;
        host_time << "Host Time(us): " << optensor.host_time;
        std::stringstream mlu_compute_time;
        mlu_compute_time << "MLU Time(us): " << optensor.mlu_compute_time;
        std::stringstream memcpy_time;
        memcpy_time << "CopyIn Time(us): " << optensor.memcpyH2D_time << "; CopyOut Time(us): " << optensor.memcpyD2H_time;

        LOG(host_time.str());
        LOG(mlu_compute_time.str());
        LOG(memcpy_time.str());
    }
    else
    {
        LOG("FAILED");
    }

    return 0;
}