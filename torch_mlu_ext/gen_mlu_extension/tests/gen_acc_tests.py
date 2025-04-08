import ast
import os
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

ORIGIN_URL = "https://pub.sakana.ai/ai-cuda-engineer/leaderboard?show_kernels=1&level=1&sort_by=level_task&experiment=all"
DOWNLOAD_DIR = "./raw_torch_functional"
MLU_SRC_DIR = "../mlu_custom_ext/src/mlu"
FWD_SRC_DIR = "../mlu_custom_ext/src"


def logical_xor(a, b):
    return bool(a) != bool(b)


def grab_torch_functional(download_dir, origin_url, mlu_dir):
    head = "https://pub.sakana.ai"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/120.0.0.0 Safari/537.36'
    }

    for mlu_file in Path(mlu_dir).glob('*.mlu'):
        task_id = mlu_file.name.split('prlblem_')[1].split('_')[0]
        print(f"Grabbing task: {task_id}")

        try:
            response = requests.get(origin_url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Cannot Access: {e}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        target = soup.find('tr', {'data-task':task_id}).find('a')
        kernel_path = target.get('href')
        if not kernel_path:
            continue

        kernel_url = urljoin(head, kernel_path)
        try:
            kernel_resp = requests.get(kernel_url, headers=headers)
            kernel_resp.raise_for_status()
        except Exception as e:
            print(f"⚠️ Cannot Open Kernel: {e}")
            continue

        kernel_soup = BeautifulSoup(kernel_resp.text, 'html.parser')
        torch_functional_blk = kernel_soup.find('div', {'id':'pytorch-functional'})\
            .find('code', {'class':'language-python'})
        
        code_lines = []  
        for element in torch_functional_blk.children:  
            if element.name == 'span':
                code_lines.append(element.get_text())  
            else:
                code_lines.append(element)  

        formatted_code = ''.join(code_lines).strip()  

        save_path = os.path.join(download_dir, f'{mlu_file.stem}.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(formatted_code)


def extract_mlu_funcname(folder_path, x): 
    target = ''
    
    for file in Path(folder_path).glob("*.h"):  
        if file.name.startswith(x):  
            match = re.match(r'custom_\d+_(.+)', file.stem)
            target = match.group(1)
            break
    return target


def split_bracket_list(s):
    s = s.strip()
    elements = []
    current = []
    depth = 0
    for char in s:
        if char == ',' and depth == 0:
            element = ''.join(current).strip()
            if element:
                elements.append(element)
            current = []
        else:
            current.append(char)
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
    if current:
        element = ''.join(current).strip()
        if element:
            elements.append(element)
    return elements


def gen_acc_tests(download_dir, fwd_dir):
    result_file_path = "./gen_unittests.py"
    header = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_mlu
import copy
import mlu_custom_ext
import unittest

class TestMLU(unittest.TestCase):
'''
    tail = f'''
if __name__ == "__main__":
    unittest.main()
'''
    
    test_cases = []

    for torch_func_file in Path(download_dir).glob("*.txt"):
        func_name = ''
        var_def = []
        cpu_inputs_raw = []
        cpu_inputs_def = []
        cpu_output_gen = []
        mlu_inputs_def = []
        mlu_output_gen = []

        with open(torch_func_file, 'r', encoding='utf-8') as f:
            content = f.read()
            task_id = torch_func_file.name.split('prlblem_')[1].split('_')[0]

            func_name = extract_mlu_funcname(fwd_dir, "custom_" + task_id)

        if (len(func_name) == 0):
            continue

        unit_head = f'''
    def test_{func_name}(self):
'''
        
        tensor_param_list = []      
        param_pattern = re.compile(r'def\s+module_fn\s*\((.*?)\)', re.DOTALL)
        matches = param_pattern.findall(content)

        if matches != []:
            params = matches[0].split(',')
            for p in params:
                if (p.find('torch.Tensor') != -1):
                    cur_p = p.split(':')[0].strip()
                    if len(cur_p) > 0:
                        tensor_param_list.append(cur_p)
                        mlu_inputs_def.append(f"        {cur_p}_mlu = {cur_p}_cpu.to(\"mlu\")")
        
        is_comment = 0
        in_function = -1    # 0: module_fn; 1: get_inputs  
        indentation_level = -1
        starts_line = False
        starts_return = False
        
        for line in content.splitlines():
            stripped_line = line.strip()
            cur_indentation_level = len(line) - len(stripped_line)

            if stripped_line.startswith("class") or \
                stripped_line.startswith("def get_init_inputs"):
                indentation_level = 100
                continue
            elif stripped_line.startswith("def module_fn"):
                indentation_level = cur_indentation_level
                in_function = 0
                continue 
            elif stripped_line.startswith("def get_inputs"):
                indentation_level = cur_indentation_level
                in_function = 1
                starts_line = False
                starts_return = False
                continue
            elif cur_indentation_level == 0 and \
                not stripped_line.startswith("import") and \
                not stripped_line.startswith("#") and \
                not stripped_line.startswith(")") and \
                len(stripped_line) != 0:
                # var def
                var_def.append(f"        {stripped_line}")
                indentation_level = 0
                in_function = -1
                continue
            

            if stripped_line.startswith("\"\"\""):
                is_comment = logical_xor(1, is_comment)
                continue

            if (not is_comment) and (cur_indentation_level > indentation_level):
                if in_function == 0 and stripped_line.find(':') == -1:
                    stripped_line = stripped_line.replace('return ', 'result_cpu = ')
                    for tp in tensor_param_list:
                        if stripped_line.find(f"{tp},") != -1:
                            stripped_line = stripped_line.replace(f"{tp},", f"{tp}_cpu,")
                        if stripped_line.find(f"{tp})") != -1:
                            stripped_line = stripped_line.replace(f"{tp})", f"{tp}_cpu)")
                    cpu_output_gen.append(f"        {stripped_line}")
                elif in_function == 1:
                    if not starts_line:
                        starts_line = True
                        if stripped_line.startswith('return'):
                            starts_return = True
                    
                    if not starts_return and stripped_line.startswith('return'):
                        in_function = -1
                        indentation_level = 100
                        continue
                    
                    if not starts_return:
                        stripped_line = stripped_line.replace(' = ', '_cpu = ')
                    cpu_inputs_raw.append(f"        {stripped_line}")

        if starts_return:
            param_init_pattern = re.compile(r'return\s+\[(.*?)\]', re.DOTALL)
            matches = param_init_pattern.findall(''.join(cpu_inputs_raw))

            if matches != []:
                params = split_bracket_list(matches[0])
                for i in range(len(params)):
                    new_def = tensor_param_list[i] + "_cpu = " + params[i]
                    cpu_inputs_def.append(f"        {new_def}")
        else:
            cpu_inputs_def = cpu_inputs_raw

        kernel_call = '\n'.join(cpu_output_gen)
        start = kernel_call.find('(')
        mlu_output_gen = f"        result_mlu = {func_name}" + kernel_call[start:]
        mlu_output_gen = mlu_output_gen.replace("_cpu", "_mlu")

        test_cases.append(unit_head)
        test_cases.extend(var_def)
        test_cases.extend(cpu_inputs_def)
        test_cases.extend(mlu_inputs_def)
        test_cases.extend(cpu_output_gen)
        test_cases.append(mlu_output_gen)
        test_cases.append("        np.testing.assert_array_almost_equal(result_mlu.cpu(), result_cpu, decimal=3)")

    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write('\n'.join(test_cases))
        f.write(tail)


if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # grab_torch_functional(DOWNLOAD_DIR, ORIGIN_URL, MLU_SRC_DIR)
    print("\n------------ All Downloaded! ------------")

    gen_acc_tests(DOWNLOAD_DIR, FWD_SRC_DIR)
    print("\n------------ All Processed! ------------")
