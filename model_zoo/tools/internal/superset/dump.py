import argparse
import sys
import os
import yaml
import json
import logging
import re
import platform
import subprocess
from pathlib import Path

from cndb.submit import submit
from cndb.params import CndbParams
from cndb.easy import get_mlu_name, dump_mlu_machine_info, dump_pt_info

logger = logging.getLogger(os.path.basename(__file__))

legacy_names = {'dlrm', 'fairseq-transformer_wmt_en_de_big', 'waveglow', 'transformer', 'tacotron2', 'mobilenet_v2', 'crnn', 'resnet50', 'vgg16', 'alexnet', 'mmcls_regnetx_800mf_8xb128_in1k', 'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5', 'resnet18', 'googlenet', 'bbn', 'shufflenet_v2_x1_5', 'resnet50v1.5', 'inceptionv2', 'vgg16_bn', 'vgg19', 'resnet101', 'ssd_resnet50', 'inceptionv3', 'swin_tiny_patch4_window7_224', 'densenet201', 'mmaction2_tsn_r50_1x1x3_75e_ucf101_rgb', 'yolov5m', 'mmcls_convnext_base_3rdparty_32xb128_in1k', 'bert_base_finetune_msra_ner', 'bert-base-cased', 'p3d', 'yolov5s', 'dan', 'yolov3', 'mmseg_fcn_unet_s5_d16_128x128_40k_stare', 'centernet_dla34_ial', 'bert', 'rfbnet', 'mmdet_yolov3_mobilenetv2_mstrain_416_300e_coco', 'ssd-vgg16', 'mmdet_ssd512_coco', 'mmdet_yolov3_d53_mstrain_416_273e_coco', 'efficientnet', 'crf', 'wavernn', 'mmdet_retinanet_r50_fpn_1x_coco', 'fasterrcnn-resnet101+fpn', 'yolov5x_v2.0', 'maskrcnn-resnet101+fpn', 'mmdet_faster_rcnn_r101_fpn_2x_coco', 'unet3d', 'deepspeech2', 'mmdet_atss_r50_fpn_1x_coco', 'mmdet_mask_rcnn_r50_fpn_2x_coco', 'lpcnet', 'retinanet', 'lidar_rcnn', 'ssd300-resnet34', 'swin_transformer_ssl-linear_evaluation', 'inception_v4', 'conformer', 'swin_transformer_ssl-pre_training', 'bert-large', 'yolov5x_v6.0', 'enet', 'ecapa_tdnn', 'pix2pix', 'pointpillar', 'bevformer', 'p3d'}

def parse_args():
    parser = argparse.ArgumentParser("tf dumper")
    parser.add_argument("-i", "--input", help="Input file path.")
    parser.add_argument("-o", "--outdir", help="Output YAML path.")
    parser.add_argument("--machine_name", help="Machine information name.")
    parser.add_argument("--code_link", help="Code link.")

    args = parser.parse_args()
    if not args.input:
        parser.print_usage(sys.stderr)
        sys.exit(-1)
    return args

def run_cmd_info(cmd):
    try:
        if platform.system() == "Linux":
            all_info = subprocess.check_output(
                                cmd, stderr=subprocess.STDOUT, shell=True).strip()
            if type(all_info) is bytes:
                all_info = all_info.decode()
            return all_info
        else:
            logger.warning(
                "unsupported platform: {}".format(platform.system()))
    except Exception as e:
        logger.error("failed to run command: {}".format(cmd))
        raise e

def get_pt_name():
    pt_name = ""
    try:
        all_info = run_cmd_info(
            "python -c '"
            "import torch_mlu;"
            "print(\"__TORCH_MLU_VERSION: {}\".format(torch_mlu.__version__));"
            "'"
        )
        for line in all_info.split("\n"):
            if "__TORCH_MLU_VERSION" in line:
                pt_name = re.search(
                    r"__TORCH_MLU_VERSION: ([\d\.]*)", line).group(1)
        return 'v'+pt_name
    except Exception as e:
        err_msg = "failed to get pt information, due to {}".format(e)
        logger.error(err_msg)
        raise e

def get_machine_name():
    try:
        cmd = 'cat /etc/lsb-release | grep "DISTRIB_DESCRIPTION"'
        os_version = run_cmd_info(cmd)
        os_name = re.search(r"DISTRIB_DESCRIPTION=\"([\w\.\s]*)", os_version).group(1)
        return os_name
    except(subprocess.CalledProcessError):
        try:
            cmd = 'cat /etc/centos-release'
            os_version = run_cmd_info(cmd)
            os_name = re.search(r"CentOS([\sA-Za-z]*)([\d\.]*)", os_version).group(1)
            os_name = "CentOS-" + os_name
            return os_name
        except Exception as e:
            logger.error("unsupported os system: only ubuntu or centos.")
            raise e

class Reader:
    def __init__(self, hard_name, code_link):
        self.hard_info = json.loads(dump_mlu_machine_info(hard_name))
        self.dev_name = get_mlu_name()
        self.code_link = code_link

    def read_line(self, line):
        """Read data from one line

        Data example:

            network:resnet50, Batch Size:256, device count:1, Precision:O0,\
            DPF mode:single, time_avg:0.511s, time_var:0.000178,\
            throughput(fps):501.2, device:MLU290, dataset:imageNet2012
        """
        data = {}
        performance = {}
        metrics = {}
        data["eval_type"] = "-"
        for field in line.split(","):
            key, value = field.strip().split(":")
            key = key.lower()
            value = value.strip()
            if key == "precision":
                data["train_type"] = value
            elif key == "eval_exec_mode":
                data["eval_type"] = value
            elif key == "dpf mode":
                data["dist_type"] = value
            elif key == "time_avg":
                performance["latency"] = value
            elif key == "time_var":
                performance["variance"] = value
            elif key == "fps" or "throughput" in key:
                performance["throughput"] = value
            elif key == "accuracy":
                metrics["accuracy"] = value
            elif key == "sw":
                metrics["sw"] = value
            elif key == "batch size":
                data["batch_size"] = int(value)
            elif key == "device count":
                data["dev_num"] = int(value)
            elif key == "network":
                data["model"] = value

            else:
                data[key] = value
        data["perf_data"] = performance
        data["framework"] = "pytorch"
        data["metric_data"] = metrics

        return data

    def dump(self, data, benchmark_dict, outdir):
        with open("./soft_info.json") as f:
            soft_info_dict = json.load(f)
        soft_info = {"name": 'v' + benchmark_dict["catch version"].strip()}
        ct_version = re.search(r"(v[\d\.]*)", soft_info["name"]).group(1)
        ct_info = soft_info_dict[ct_version]
        soft_info["ctr_version"] = ct_info["ctr_version"]
        soft_info["release_date"] = ct_info["release_date"]
        soft_info["pt"] = "1.6.0" if benchmark_dict["catch version"].strip()[-8:] == "torch1.6" else "1.9.0"

        data["code_link"] = self.code_link
        data["soft_info"] = soft_info
        data["hard_info"] = self.hard_info
        data["dev"] = data["device"] if "device" in data else self.dev_name
        outfile = os.path.join(outdir, "{}_torch_{}_batch_size_{}_dev_num_{}_precision_{}_dpf_mode_{}_device_{}.yaml".format(
          data["model"], soft_info["pt"], data["batch_size"], data["dev_num"], data["train_type"], data["dist_type"], data["dev"]))
        data["save_file"] = outfile
        print(data)
        try:
            # cndb params check
            submit(CndbParams(data))
        except Exception as e:
            logger.error("failed to dump data to {}, due to {}".format(outfile, e))

    def read_and_dump(self, infile, outdir):
        for dir0 in os.listdir(infile):
            for dir1 in os.listdir(os.path.join(infile, dir0)):
                data, performance = {}, {}
                log_file = os.path.join(infile, dir0, dir1, "benchmark_log")
                if Path(log_file).exists():
                    with open(log_file, "r") as f:
                        benchmark_dict = json.load(f)
                        performance["throughput"] = benchmark_dict["throughput"]
                        data["perf_data"] = performance
                        data["dist_type"] = benchmark_dict["DPF_mode"].strip().lower()
                        data["batch_size"] = benchmark_dict["batch_size"]
                        data["train_type"] = "mixed" if benchmark_dict["precision"] == "amp" else benchmark_dict["precision"].strip().lower()
                        data["dev_num"] = int(benchmark_dict["cards"])
                        data["device"] = benchmark_dict["device"].strip()
                        data["model"] = benchmark_dict["net"].lower().strip() if benchmark_dict["net"].lower().strip() in legacy_names else benchmark_dict["net"].strip()
                        data["dataset"] = benchmark_dict["dataset"].strip()
                        if len(data["dataset"]) == 0:
                            data["dataset"] = "unknow"
                        data["framework"] = "pytorch"
                    self.dump(data, benchmark_dict, outdir)
                else:
                    logger.warning("unknow benchmark_log file: {}!!!!!!!!!!".format(log_file))

if __name__ == "__main__":
    args = parse_args()
    if args.machine_name is None:
        args.machine_name = get_machine_name()
    reader = Reader(args.machine_name, args.code_link)
    reader.read_and_dump(args.input, args.outdir)
