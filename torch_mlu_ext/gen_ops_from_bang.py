import os
import re
from pathlib import Path
import argparse 


def get_func_body(content):
    stack = []  
    start_index = -1  
    end_index = -1  
    
    for index, char in enumerate(content):  
        if char == '{':  
            if not stack:
                start_index = index  
            stack.append(char)  
        elif char == '}':  
            if stack:  
                stack.pop()  
            if not stack:
                end_index = index  
                break  
                
    if start_index != -1 and end_index != -1:  
        return content[start_index + 1:end_index]  
    
    return None 


def split_parameters(param_list):   
    params = []  
    current_param = []  
    stack = []

    for char in param_list:  
        if char == '<':
            stack.append(char)  
        elif char == '>':
            if stack:  
                stack.pop()  
        elif char == ',' and not stack: 
            params.append(''.join(current_param).strip())
            current_param = []
            continue

        current_param.append(char) 

    if current_param:  
        params.append(''.join(current_param).strip())  

    return params  


def process_mlu_files(MLU_INPUT_DIR, BANG_DIR, kernels_header_file):
    kernel_pattern = re.compile(
        r'__mlu_global__\s+void\s+(\w+)\(([^)]*)\)\s*{',
        re.MULTILINE
    )

    for mlu_file in Path(BANG_DIR).glob('*.mlu'):
        print(f"Processing {mlu_file.name}...")
        
        with open(mlu_file, 'r+', encoding='utf-8') as f:
            content = f.read()

        if ((content.count('(') != content.count(')')) or (content.count('{') != content.count('}'))):
            continue

        with open(os.path.join(MLU_INPUT_DIR, f'{mlu_file.stem}.mlu'), 'w', encoding='utf-8') as f:
            f.write(content)
            
            matches = kernel_pattern.findall(content)
            if not matches:
                print(f"No kernel found in {mlu_file}")
                continue
                
            entry_code = []
            header_decls = []
            
            for func_name, params in matches:
                if params == '':
                    params_ = "int elem_num"
                else:
                    params_ = params + ", int elem_num"

                entry_func = f"void {func_name}_entry({params_})"
                header_decls.append(f"{entry_func};")

                params_list = split_parameters(params)
                params_name_list = []

                for p in params_list:
                    last_space_index = p.rfind(' ')
                    name = p[last_space_index + 1:].strip()
                    params_name_list.append(name)
                
                entry_code.append(f"\n// Auto-generated entry function for {func_name}")
                entry_code.append(entry_func + " {")
                entry_code.append("    cnrtQueue_t queue;")
                entry_code.append("    cnrtQueueCreate(&queue);")
                entry_code.append("    cnrtDim3_t dim = {1, 1, 1};")
                entry_code.append("    cnrtFunctionType_t c = CNRT_FUNC_TYPE_BLOCK;")
                entry_code.append(f"    dim.x = elem_num / 32;")
                entry_code.append(f"    {func_name}<<<dim, c, queue>>>({', '.join(p for p in params_name_list)});")
                entry_code.append("    cnrtQueueSync(queue);")
                entry_code.append("    cnrtQueueDestroy(queue);")
                entry_code.append("}\n")
            
            f.seek(0, 2) 
            f.write("\n\n// ********** Entry Functions **********\n")
            f.write('\n'.join(entry_code))
            
        with open(kernels_header_file, 'a', encoding='utf-8') as hf:
            hf.write('\n'.join(header_decls))
            hf.write("\n")
            
        print(f"    Generated {header_decls}")


def process_fwd_codes(FWD_INPUT_DIR, FWD_FINAL_DIR, MLU_INPUT_DIR, REG_DIR):
    for mlu_file in Path(MLU_INPUT_DIR).glob('level_1_*.mlu'):
        txt_path = os.path.join(FWD_INPUT_DIR, f'{mlu_file.stem}.txt') 

        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content_ = content.replace("cudaDeviceSynchronize", "//cudaDeviceSynchronize")
        content_ = content_.replace("TORCH_CHECK", "//TORCH_CHECK")
        content_ = content_.replace("INPUT_CHECK", "//INPUT_CHECK")
        content_ = content_.replace("dim3", "//dim3")
        # content_ = content_.replace("int ", "int64_t ")
        # content_ = content_.replace("float ", "double ")

        func_match = re.search(
            r'torch::Tensor\s+(\w+)\(([^)]*)\)\s*{',
            content_,
            re.MULTILINE
        )
        if not func_match:
            print(f"Func Decl Not Found: {txt_path}")
            continue

        func_name = func_match.group(1).replace("cuda", "mlu")
        params = func_match.group(2)
        body = get_func_body(content_)
        task_id = mlu_file.name.split('prlblem_')[1].split('_')[0]

        cpp_filename = f"/custom_{task_id}_{func_name}.cpp"
        cpp_path = FWD_FINAL_DIR + cpp_filename
        header_filename = f"/custom_{task_id}_{func_name}.h"
        header_path = FWD_FINAL_DIR + header_filename

        # Add-on dev
        if os.path.isfile(cpp_path):
            continue

        tensor_params = []
        register_params = ''
        register_names = ''
        param_lines = split_parameters(params)

        for p in param_lines:
            last_space_index = p.rfind(' ')
            type = p[:last_space_index].strip()
            name = p[last_space_index + 1:].strip()
            if type == 'torch::Tensor':
                type = 'Tensor'
            register_params += (f"{name}:{type},")
            register_names += (f"{name},")

            if p.startswith('torch::Tensor'):
                param_name = p.split()[-1]
                tensor_params.append(param_name)

        print(f"Registering {func_name} ...")

        with open(f"{REG_DIR}/__init__.py", 'a', encoding='utf-8') as f:
            f.write(f"\nfrom .custom_ops import {func_name}")

        with open(f"{REG_DIR}/custom_ops.py", 'a', encoding='utf-8') as f:
            f.write(f"\n\ndef {func_name}({register_params}):")
            f.write(f"\n    return torch.ops.mlu_custom_ext.{func_name}({register_names})")

        print(f"Processing fwd function {func_name}...")

        # Start to Trans
        header = f'''#include "{header_filename[1:]}"
#include "mlu/gen_kernels.h"
#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;
'''
        modified_body = []
        kernel_call_body = []
        tensor_param_init = []
        new_tensor_init = []
        first_inp = ""

        tensor_param_init.append(f'torch::Tensor {func_name}({params}) {{')
        tensor_param_init.append(f'    const torch_mlu::mlu::MLUGuard device_guard({tensor_params[0]}.device());')

        for param in tensor_params:
            if first_inp == "":
                first_inp = param
            tensor_param_init.extend([
                f'    auto {param}_contiguous = torch_mlu::cnnl_contiguous({param});',
                f'    auto {param}_impl = getMluTensorImpl({param}_contiguous);',
                f'    auto {param}_ptr = {param}_impl->mlu_data_ptr();'
            ])

        current_block = "" 
        kernel_start = False 
        in_block = False
        new_tensor_pattern = r'(\w+)\.data_ptr<float>\(\)'

        for line in body.split('\n'):  
            line = re.sub(r'\btorch::(\w+)', r'at::\1', line)  

            if '<<<' in line:  
                in_block = True
                kernel_start = True  
                current_block += line 
                
                if '>>>' in line:  
                    line = re.sub(r'(\w+)<<<(.*?)>>>\s*\(', r'\1_entry(', current_block, flags=re.DOTALL)  
                    current_block = ""  
                    in_block = False  
                else:  
                    continue 

            elif in_block:  
                current_block += line  
                if '>>>' in line:  
                    line = re.sub(r'(\w+)<<<(.*?)>>>\s*\(', r'\1_entry(', current_block, flags=re.DOTALL)  
                    current_block = ""  
                    in_block = False  
                    
            for param in tensor_params:
                line = re.sub(rf'\b{param}\b', f'{param}_contiguous', line)
            line = re.sub(r'(\w+)_contiguous\.data_ptr<float>\(\)', r'reinterpret_cast<float*>(\1_ptr)', line)

            new_tensors = re.findall(new_tensor_pattern, line)
            for new_tensor in new_tensors:
                new_tensor_init.extend([
                    f'    auto {new_tensor}_contiguous = torch_mlu::cnnl_contiguous({new_tensor});',
                    f'    auto {new_tensor}_impl = getMluTensorImpl({new_tensor}_contiguous);',
                    f'    auto {new_tensor}_ptr = {new_tensor}_impl->mlu_data_ptr();'
                ])
            
            line = re.sub(r'(\w+)\.data_ptr<float>\(\)', r'reinterpret_cast<float*>(\1_ptr)', line)
            
            if kernel_start:
                if line.find("();") != -1:
                    line = line.replace(");", "elem_num);")
                else:
                    line = line.replace(");", ", elem_num);")
                kernel_call_body.append(f"    {line.strip()}")
            else:
                modified_body.append(f"    {line.strip()}")  


        processed_code = [header]
        processed_code.extend(tensor_param_init)
        processed_code.extend(modified_body)
        processed_code.extend(new_tensor_init)
        processed_code.append(f'    auto elem_num = {first_inp}_contiguous.numel();')
        processed_code.extend(kernel_call_body)
        processed_code.append('}\n')

        processed_params = re.sub(r'torch::Tensor\s+', 'Tensor ', params)
        processed_code.append(f'''
TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {{
    m.def("{func_name}({processed_params}) -> Tensor");
}}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {{
    m.impl(
        TORCH_SELECTIVE_NAME("mlu_custom_ext::{func_name}"),
        TORCH_FN({func_name}));
}}
    ''')

        with open(cpp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_code))
        print(f"    Generated fwd: {cpp_path}")

        with open(header_path, 'w', encoding='utf-8') as f:
            f.write("#pragma once\n")
            f.write("#include <torch/extension.h>\n")
            f.write(f'torch::Tensor {func_name}({params});')
        print(f"    Generated fwd: {header_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--BANG_DIR', type=str, default="./gen_mlu_extension/mlu_custom_ext/src/mlu", help='原始 .mlu 文件所在目录')
    parser.add_argument('--MLU_INPUT_DIR', type=str, default="./gen_mlu_extension/mlu_custom_ext/src/mlu", help='修改后 .mlu 文件存储目录')
    parser.add_argument('--FWD_INPUT_DIR', type=str, default="../cuda_ops_test/Ascend_kernels/gen_fwd_codes", help="源文件 forward code 导出目录")
    parser.add_argument('--FWD_FINAL_DIR', type=str, default="./gen_mlu_extension/mlu_custom_ext/src", help="torch_mlu forward code 生成目录")
    parser.add_argument('--REG_DIR', type=str, default="./gen_mlu_extension/mlu_custom_ext/ops", help="torch_mlu 算子注册目录")
    
    args = parser.parse_args() 
    print(f"BANG_DIR: {args.BANG_DIR}")
    print(f"MLU_INPUT_DIR: {args.MLU_INPUT_DIR}")  
    print(f"FWD_INPUT_DIR: {args.FWD_INPUT_DIR}") 
    print(f"FWD_FINAL_DIR: {args.FWD_FINAL_DIR}") 
    print(f"REG_DIR: {args.REG_DIR}") 

    os.makedirs(args.MLU_INPUT_DIR, exist_ok=True)
    os.makedirs(args.FWD_INPUT_DIR, exist_ok=True)
    os.makedirs(args.FWD_FINAL_DIR, exist_ok=True)
    os.makedirs(args.REG_DIR, exist_ok=True)
     
    kernels_header_file = os.path.join(args.MLU_INPUT_DIR, 'gen_kernels.h') 
    with open(kernels_header_file, 'w') as f:  
        f.write("#pragma once\n")
        f.write("#include <cnrt.h>\n\n")
        f.write("// Auto-generated declarations\n")

    print("\nGenerating entry code for MLU Kernels...")
    if process_mlu_files(args.MLU_INPUT_DIR, args.BANG_DIR, kernels_header_file):
        print("\nDone")

    print("\nGenerating forward code for Ops...")
    with open(f"{args.REG_DIR}/__init__.py", 'a', encoding='utf-8') as f:
        f.truncate(0)

    with open(f"{args.REG_DIR}/custom_ops.py", 'a', encoding='utf-8') as f:
        f.truncate(0)
        f.write(f"\nimport torch")
        f.write(f"\nfrom torch import Tensor\n")
    if process_fwd_codes(args.FWD_INPUT_DIR, args.FWD_FINAL_DIR, args.MLU_INPUT_DIR, args.REG_DIR):
        print("\nDone")
    print("\nProcessing completed!")