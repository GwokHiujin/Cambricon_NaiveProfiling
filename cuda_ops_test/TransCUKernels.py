import os
import re
from pathlib import Path
import argparse 

def generate_cu_files(file_path, CUDA_OUTPUT_DIR, FWD_OUTPUT_DIR):
    other_patterns = [
        (r'^(__device__\s+\w+.*?)\s*$', 1),
        (r'^(__global__\s+\w+.*?)\s*$', 1),
        (r'^(#include\s*<\w+.*?)\s*$', 0), 
        (r'^(#define\s+\w+.*?)\s*$', 0)
    ]

    fwd_pattern = [
        (r'^(torch::Tensor\s+\w+.*?)\s*$'),
        (r'^(std::vector<torch::Tensor>\s+\w+.*?)\s*$'),
    ]

    decl_pattern = (r'torch::Tensor\s+(\w+)\(([^)]*)\)\s*\;')

    base_name = Path(file_path).stem
    output_path = CUDA_OUTPUT_DIR + f"/{base_name}.cu"
    fwd_path = FWD_OUTPUT_DIR + f"/{base_name}.txt"

    with open(file_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        f.seek(0)

        dev_code_block = False
        brace_level = 0
        first_left_brace = False
        first_write = True
        is_fwd = False
        function_lines = []
        forward_lines = []
        
        for line in lines:                
            if not dev_code_block:
                if re.match(decl_pattern, line, re.MULTILINE):
                    continue

                for pattern in fwd_pattern: 
                    if re.match(pattern, line, re.MULTILINE):
                        is_fwd = True
                        dev_code_block = True
                        first_left_brace = False
                        brace_level = 1
                        break
                    
                for pattern, start_level in other_patterns:
                    if re.match(pattern, line, re.MULTILINE) and \
                                ("<torch" not in line) and \
                                ("<pybind" not in line):
                        dev_code_block = True
                        first_left_brace = False
                        brace_level = start_level
                        break
            
            if dev_code_block:
                open_braces = line.count('{')
                close_braces = line.count('}')
                if first_left_brace == False and open_braces > 0:
                    first_left_brace = True
                    open_braces -= 1
                brace_level += (open_braces - close_braces)
                
                if is_fwd:
                    forward_lines.append(line.rstrip('\n'))
                else:
                    function_lines.append(line.rstrip('\n'))
                
                if brace_level <= 0 and first_left_brace:
                    is_fwd = False
                    dev_code_block = False
                    brace_level = 0

                    with open(output_path, 'w' if first_write else 'a', encoding='utf-8') as out_f:
                        out_f.write('\n'.join(function_lines))
                        out_f.write('\n') 
                    with open(fwd_path, 'w' if first_write else 'a', encoding='utf-8') as fwd_f:
                        fwd_f.write('\n'.join(forward_lines))
                        fwd_f.write('\n')

                    first_write = False
                    function_lines = []
                    forward_lines = []

        print(f"Generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--SRC_DIR', type=str, default="./Ascend_kernels", help='源代码所在目录')
    parser.add_argument('--CUDA_OUTPUT_DIR', type=str, default="./Ascend_kernels/gen_cuda_kernels", help='期望存储输出的 CUDA CODE 的目录')
    parser.add_argument('--FWD_OUTPUT_DIR', type=str, default="./Ascend_kernels/gen_fwd_kernels", help='期望存储输出的 forward code 的目录')


    args = parser.parse_args()  
    print(f"SRC_DIR: {args.SRC_DIR}")  
    print(f"CUDA_OUTPUT_DIR: {args.CUDA_OUTPUT_DIR}") 
    print(f"FWD_OUTPUT_DIR: {args.FWD_OUTPUT_DIR}") 

    os.makedirs(args.CUDA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(args.FWD_OUTPUT_DIR, exist_ok=True)

    ascend_files = Path(args.SRC_DIR).glob('*.py')
    for ascend_file in ascend_files:
        print(f"🔗 Processing: {ascend_file.name}")
        generate_cu_files(ascend_file, args.CUDA_OUTPUT_DIR, args.FWD_OUTPUT_DIR)