# 从 CUDA Code 到 MLU 注册算子工作流

仅适用于 forward 函数跟 cuda kernel 写在同一个文件内、需要批量注册为 torch_mlu 算子的场景。

参考目录：

- SRC_DIR: /root/Cambricon_NaiveProfiling/cuda_ops_test/Ascend_kernels
- FWD_OUTPUT_DIR: /root/Cambricon_NaiveProfiling/cuda_ops_test/Ascend_kernels/gen_fwd_codes
- CUDA_OUTPUT_DIR: /root/Cambricon_NaiveProfiling/cuda_ops_test/Ascend_kernels/gen_cuda_kernels
- CU_SRC_DIR: $CUDA_OUTPUT_DIR
- BANG_DIR: /root/Cambricon_NaiveProfiling/cuda_ops_test/Ascend_kernels/gen_bang_results
- MLU_INPUT_DIR: /root/Cambricon_NaiveProfiling/torch_mlu_ext/gen_mlu_extension/mlu_custom_ext/src/mlu
- FWD_INPUT_DIR: $FWD_OUTPUT_DIR
- FWD_FINAL_DIR: /root/Cambricon_NaiveProfiling/torch_mlu_ext/gen_mlu_extension/mlu_custom_ext/src
- REG_DIR: /root/Cambricon_NaiveProfiling/torch_mlu_ext/gen_mlu_extension/mlu_custom_ext/ops

---


1. 在 `Cambricon_NaiveProfiling/cuda_ops_test/TransCUKernels.py` 脚本中指定源代码目录 `SRC_DIR`、期望存储分离的 Cuda Code 的目录 `CUDA_OUTPUT_DIR` 和期望存储分离的 Forward Code 的目录 `FWD_OUTPUT_DIR`，并运行该脚本。

```bash
/torch/venv3/pytorch/bin/python TransCUKernels.py --SRC_DIR $SRC_DIR --CUDA_OUTPUT_DIR $CUDA_OUTPUT_DIR --FWD_OUTPUT_DIR $FWD_OUTPUT_DIR
```

这一步可将源代码中的 Cuda Kernel Code 分离成独立的 .cu 文件。

2. 在 `Cambricon_NaiveProfiling/cuda_ops_test/gen_bang_code.sh` 脚本中指定 CUDA Code 源代码目录 `CU_SRC_DIR` 和期望存储生成的 BANG Code 的目录 `BANG_DIR`，并运行该脚本。

```bash
bash gen_bang_code.sh $CU_SRC_DIR $BANG_DIR
```

这一步可将 CUDA Code 转换为 BANG Code。

> 可能存在转换失败的例子。

3. 在 `Cambricon_NaiveProfiling/torch_mlu_ext/gen_ops_from_bang.py` 脚本中指定：

- BANG_DIR：.mlu 源文件所在目录，一般与第 2 步的 BANG_DIR 为同一目录
- MLU_INPUT_DIR：期望存储 MLU 算子源码的目录
- FWD_INPUT_DIR：分离的原始 Forward Code 所在目录
- FWD_FINAL_DIR：期望存储生成的 Forward Code 的目录

```bash
/torch/venv3/pytorch/bin/python gen_ops_from_bang.py --BANG_DIR $BANG_DIR \
    --MLU_INPUT_DIR $MLU_INPUT_DIR \
    --FWD_INPUT_DIR $FWD_INPUT_DIR \
    --FWD_FINAL_DIR $FWD_FINAL_DIR \
    --REG_DIR $REG_DIR
```

这一步可将源代码中的 Forward Code 分离成独立的 .txt 文件，并根据该文件的内容生成注册 torch_mlu 算子需要的各种文件。

> 可能存在转换失败的例子，例如某些 forward code 中包含一些 torch 不直接支持的数据类型，可以查看 `Cambricon_NaiveProfiling/torch_mlu_ext/mlu_extension/mlu_custom_ext/ops/custom_ops.py` 里的参数类型手动修改。

4. 注册全部的 torch_mlu 算子

```bash
cd /root/Cambricon_NaiveProfiling/torch_mlu_ext/gen_mlu_extension
python setup.py install
```

---

正确性测试代码需手动实现。
