#ifndef _BANGC_COMMON_H
#define _BANGC_COMMON_H

#include <cn_api.h>
#include <cndev.h>

#define ELEM_NUM (16 * 256 * 112 * 112)
#define NFU_ALIGN_SIZE 128

#if (__BANG_ARCH__ == 372)
#define MAX_NRAM_SIZE 655360
#define MAX_SRAM_SIZE 4063232
#else // mtp_592
#define MAX_NRAM_SIZE 327680
#define MAX_SRAM_SIZE 1966080
#endif

#define CNDRV_CHECK(val)                                                       \
  do {                                                                         \
    CNresult ret = val;                                                        \
    if (ret) {                                                                 \
      const char *error_string;                                                \
      cnGetErrorString(ret, &error_string);                                    \
      printf("[%s:%d] CNDRV error, code=%u(%s) \"%s\" \n", __FILE__, __LINE__, \
             (unsigned int)ret, error_string, #val);                           \
      exit(-1);                                                                \
    }                                                                          \
  } while(0)

#define CNDEV_CHECK(val)                                                       \
  do {                                                                         \
    cndevRet_t ret = val;                                                      \
    if (ret) {                                                                 \
      printf("[%s:%d] CNDEV error, code=%u(%s) \"%s\" \n", __FILE__, __LINE__, \
             (unsigned int)ret, cndevGetErrorString(ret), #val);               \
      exit(-1);                                                                \
    }                                                                          \
  } while(0)

inline int32_t get_core_num_per_cluster() {
  CNdev mlu_dev;
  CNDRV_CHECK(cnCtxGetDevice(&mlu_dev));
  int32_t core_num_per_cluster = -1;
  CNDRV_CHECK(
    cnDeviceGetAttribute(&core_num_per_cluster,
                         CN_DEVICE_ATTRIBUTE_MAX_CORE_COUNT_PER_CLUSTER,
                         mlu_dev));
  return core_num_per_cluster;
}

inline int32_t get_cluster_num() {
  CNcontext drv_ctx;
  CNctxConfigParam ctx_conf_param;
  CNDRV_CHECK(cnCtxGetCurrent(&drv_ctx));
  CNDRV_CHECK(cnGetCtxConfigParam(drv_ctx,
                                  CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM,
                                  &ctx_conf_param));
  return (int32_t)ctx_conf_param.visibleClusterNumber;
}

// get peak compute force, gflops
inline double get_peak_compute_force() {
  int card = -1;
  int ipu_frequency = -1;
  CNRT_CHECK(cnrtGetDevice(&card));
  CNDEV_CHECK(cndevInit(0));
  CNDRV_CHECK(cnDeviceGetAttribute(&ipu_frequency,
                                   CN_DEVICE_ATTRIBUTE_CLUSTER_CLOCK_RATE,
                                   card));
  cndevDeviceMaxPerformance_t max_performance;
  max_performance.version = CNDEV_VERSION_5;
  cndevGetDeviceMaxPerformance(&max_performance, card);

  // only addition, not including multiplication
  uint64_t peak_compute_force_f32_add = max_performance.fp32Vector / 2;
  return peak_compute_force_f32_add * get_cluster_num() *
           get_core_num_per_cluster() * (double)ipu_frequency / 1000 / 1000;
}

// get io bandwidth, GB/s
inline double get_io_bandwidth() {
  int card = -1;
  CNRT_CHECK(cnrtGetDevice(&card));
  CNDEV_CHECK(cndevInit(0));
  cndevDDRInfo_t ddrinfo;
  ddrinfo.version = CNDEV_VERSION_5;
  CNDEV_CHECK(cndevGetDDRInfo(&ddrinfo, card));
  double band_width = ddrinfo.bandWidth;
  double band_width_decimal = ddrinfo.bandWidthDecimal;
  do {
    band_width_decimal /= 10;
  } while (band_width_decimal > 1);

  double result = band_width + band_width_decimal;
  return result;
}

inline void get_policy_function_block(cnrtDim3_t *dim, cnrtFunctionType_t *func_type) {
  *func_type = cnrtFuncTypeBlock;
  dim->x = 1;
  dim->y = 1;
  dim->z = 1;
  return;
}

inline void get_policy_function_union1(cnrtDim3_t *dim, cnrtFunctionType_t *func_type) {
  *func_type = cnrtFuncTypeUnion1;
  dim->x = get_core_num_per_cluster();
  dim->y = get_cluster_num();
  dim->z = 1;
  return;
}

#endif // _BANGC_COMMON_H