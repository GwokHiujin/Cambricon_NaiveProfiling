import os
import re
from pathlib import Path

ASCEND_DIR = "./Ascend_kernels"
OUTPUT_DIR = ASCEND_DIR + f"/gen_cuda_kernels"

def generate_cu_files(file_path):
    patterns = [
        (r'^(__device__\s+\w+.*?)\s*$', 1),
        (r'^(__global__\s+\w+.*?)\s*$', 1),
        (r'^(#include\s*<\w+.*?)\s*$', 0), 
        (r'^(#define\s+\w+.*?)\s*$', 0)
    ]

    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)

            dev_code_block = False
            brace_level = 0
            first_left_brace = False
            function_lines = []
            
            for line in lines:                
                if not dev_code_block:
                    for pattern, start_level in patterns:
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
                    
                    function_lines.append(line.rstrip('\n'))
                    
                    if brace_level <= 0:
                        dev_code_block = False
                        brace_level = 0

            base_name = Path(file_path).stem
            output_path = OUTPUT_DIR + f"/{base_name}.cu"
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write('\n'.join(function_lines))
                out_f.write('\n') 
                        
            print(f"Generated successfully: {file_path}")
    except Exception as e: 
        print(f"❌ Fail at {file_path} : {str(e)}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ascend_files = Path(ASCEND_DIR).glob('*.py')
    for ascend_file in ascend_files:
        print(f"🔗 Processing: {ascend_file.name}")
        generate_cu_files(ascend_file)