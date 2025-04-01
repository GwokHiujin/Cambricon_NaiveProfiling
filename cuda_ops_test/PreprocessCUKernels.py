import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_cuda_kernels(origin_url, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    head = "https://pub.sakana.ai"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(origin_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Cannot Access: {e}")
        return
    
    time.sleep(5)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    kernel_links = soup.find_all('a', {'class': 'kernel-name'})

    for index, link in enumerate(kernel_links):
        kernel_path = link.get('href')
        if not kernel_path:
            continue

        kernel_url = urljoin(head, kernel_path)
        print(f"\n🔗 Processing Kernel: ({index+1}/{len(kernel_links)}): {kernel_url}")

        try:
            kernel_resp = requests.get(kernel_url, headers=headers)
            kernel_resp.raise_for_status()
        except Exception as e:
            print(f"⚠️ Cannot Open Kernel: {e}")
            continue

        kernel_soup = BeautifulSoup(kernel_resp.text, 'html.parser')
        download_btns = kernel_soup.find_all('a', {
            'class': 'download-button'
        })

        for download_btn in download_btns:
            download_path = download_btn.get('href')
            if download_path.startswith("/ai-cuda-engineer/download/cuda/"):
                download_url = urljoin(head, download_path)
                try:
                    file_resp = requests.get(download_url, headers=headers)
                    file_resp.raise_for_status()
                except Exception as e:
                    print(f"⚠️ Download Failed: {e}")
                    continue
            else:
                continue

            if 'Content-Disposition' in file_resp.headers:
                filename = file_resp.headers['Content-Disposition'].split('filename=')[-1].strip('"')
            else:
                filename = os.path.basename(download_path) or f"kernel_{index}.cu"
                if not filename.endswith('.cu'):
                    filename += '.cu'

            save_path = os.path.join(download_dir, filename)
            with open(save_path, 'wb') as f:
                f.write(file_resp.content)
            print(f"✅ Saved: {filename}")


import re
from pathlib import Path

def sanitize_cuda_code(file_path):
    patterns = [
        (r'^#include\s*<torch/extension\.h>\s*$', 0),
        (r'^#include\s*<pybind11/pybind11\.h>\s*$', 0),
        (r'^(namespace\s+\w+.*?)\s*$', 0),
        (r'^(torch::Tensor\s+\w+.*?)\s*$', 1),
        (r'^(at::Tensor\s+\w+.*?)\s*$', 1),
        (r'^PYBIND11_MODULE\s*\(.*?\)\s*{', 1)
    ]

    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)
            
            comment_block = False
            brace_level = 0
            first_left_brace = False
            
            for line in lines:
                line = line.rstrip('\n')
                
                if not comment_block:
                    for pattern, start_level in patterns:
                        if re.match(pattern, line, re.MULTILINE):
                            comment_block = True
                            first_left_brace = False
                            brace_level = start_level
                            line = f'// {line}'
                            break
                
                if comment_block:
                    open_braces = line.count('{')
                    close_braces = line.count('}')
                    if first_left_brace == False and open_braces > 0:
                        first_left_brace = True
                        open_braces -= 1
                    brace_level += (open_braces - close_braces)
                    
                    if not line.startswith('//'):
                        line = f'// {line}'
                    
                    if brace_level <= 0:
                        comment_block = False
                        brace_level = 0
                
                f.write(f'{line}\n')
            
            f.truncate()
            
    except Exception as e:
        print(f"❌ Fail: {str(e)}")


def preprocess_cuda_files(directory):
    cuda_files = Path(directory).glob('*.cu')
    
    for cuda_file in cuda_files:
        print(f"🔗 Processing: {cuda_file.name}")
        sanitize_cuda_code(cuda_file)


if __name__ == "__main__":
    ORIGIN_URL = "https://pub.sakana.ai/ai-cuda-engineer/leaderboard?show_kernels=1&level=1&sort_by=level_task&experiment=all"
    DOWNLOAD_DIR = "./cuda_ops"  

    # download_cuda_kernels(ORIGIN_URL, DOWNLOAD_DIR)
    print("\n------------ All Downloaded! ------------")
    preprocess_cuda_files(DOWNLOAD_DIR)
    print("\n------------ All Processed! ------------")