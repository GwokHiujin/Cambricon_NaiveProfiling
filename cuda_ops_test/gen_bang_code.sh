PROJECT_ROOT_DIR=/CUDA2BANG/cuda2bang
POLYGEIST_BUILD_DIR=$PROJECT_ROOT_DIR/polygeist/build
LLVM_BUILD_DIR=$PROJECT_ROOT_DIR/polygeist/mlir-build
CGEIST=$POLYGEIST_BUILD_DIR/bin/cgeist
CUDA_GPU_ARCH=sm_70
CU_SRC_DIR=./cuda_ops
BANG_DIR=./gen_bang_results

export POLYGEIST_GPU_KERNEL_COARSEN_THREADS=1
export POLYGEIST_GPU_KERNEL_COARSEN_BLOCKS=1
export POLYGEIST_GPU_KERNEL_BLOCK_SIZE=32
export POLYGEIST_VECTORIZE_SIZE=1

# DEBUG_DIALECAT_CONVERSION=-debug-only=dialect-conversion
DEBUG_DIALECAT_CONVERSION=

mkdir $BANG_DIR

total_files=$(find "$CU_SRC_DIR" -name '*.cu' | wc -l)  
current_file=0  
failed_files=()

for cu_file in "$CU_SRC_DIR"/*.cu; do  
    ((current_file++))  

    filename=$(basename -- "$cu_file")  
    filename_noext="${filename%.cu}"  

    progress=$((current_file * 100 / total_files))  

    echo "[Generating Bang Code] $filename_noext ($current_file / $total_files)" 
    
    if $CGEIST --function=* -cuda-lower -output-intermediate-gpu -scal-rep=0 \
        -raise-scf-to-affine --cuda-gpu-arch=$CUDA_GPU_ARCH -parallel-licm=1 \
        -gpu-kernel-structure-mode=block_thread_noops --enable-buffer-elim=0 -O2 \
        -I$LLVM_BUILD_DIR/projects/openmp/runtime/src/ -resource-dir=$LLVM_BUILD_DIR/lib/clang/18/ \
        -I$LLVM_BUILD_DIR/projects/openmp/runtime/src/ -I/usr/local/cuda/include/ \
        -use-original-gpu-block-size --emit-npu=distribute.mincut -use-my-pass \
        $DEBUG_DIALECAT_CONVERSION -bang-dump-file=$BANG_DIR/$filename_noext.mlu "$cu_file" \
        -o "$filename_noext.o" > "$BANG_DIR/$filename_noext-output.txt" 2>&1; then
        echo "Done"
    else
        echo "Fail"
        failed_files+=("$filename_noext")
    fi

done  

echo "\n------------ All files processed ------------\n" 

if [ ${#failed_files[@]} -gt 0 ]; then  
    echo "The following files failed to process, please check:"  
    for failed_file in "${failed_files[@]}"; do  
        echo "$failed_file"
    done 
fi 
