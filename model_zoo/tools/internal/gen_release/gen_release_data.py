from collections import OrderedDict
import pandas as pd
import argparse
import sys
import os
import json

NET_WORK_MAPPING = OrderedDict([('resnet50', 'ResNet50_v1.5'),
                    ('resnet18', 'ResNet18'), ('ngc_resnet50v1_5', 'NGC_ResNet50-v1.5'),
                    ('alexnet', 'AlexNet'), ('vgg16', 'VGG16'),
                    ('vgg16_bn', 'VGG16_bn'), ('inceptionv3', 'InceptionV3'),
                    ('mobilenet_v2', 'MobileNetV2'), ('googlenet', 'GoogleNet'),
                    ('resnet101', 'ResNet101'), ('vgg19', 'VGG19'),
                    ('inceptionv2', 'InceptionV2'), ('shufflenet_v2_x0_5', 'Shufflenet-v2_x0_5'),
                    ('shufflenet_v2_x1_0', 'Shufflenet-v2_x1_0'), ('shufflenet_v2_x1_5', 'Shufflenet-v2_x1_5'),
                    ('densenet201', 'DenseNet201'), ('Fairseq', 'Transformer-fairseq'),
                    ('transformer', 'Transformer'), ('bert-base-cased', 'ModelZoo-BERT'),
                    ('BERT', 'NGC-BERT'), ('bert_base_finetune_msra_ner', 'Bert_base_fine-tune_msra_ner'),
                    ('SSD_Resnet50', 'SSD-ResNet50'), ('SSD_VGG16', 'SSD-VGG16'),
                    ('yolov3', 'YOLOv3'), ('yolov5s', 'YOLOv5-s'),
                    ('yolov5x', 'YOLOv5-x'),
                    ('yolov5m', 'YOLOv5-m'), ('retinanet', 'RetinaNet'),
                    ('MaskRCNN-ResNet101+FPN', 'MaskRCNN-ResNet101+FPN'), ('FasterRCNN-ResNet101+FPN', 'FasterRCNN-ResNet101+FPN'),
                    ('centernet_dla_ial', 'CenterNet'), ('crnn', 'CRNN'), ('fairseq-transformer_wmt_en_de_big','Fairseq-transformer_wmt_en_de_big'),
                    ('mmseg_unet2d','Unet-S5-D16-FCN'),('mmaction_tsn', 'MMAction-TSN2D'),
                    ('mmdet_maskrcnn', 'MMDet_Mask R‑CNN'),('mmdet_ssd','MMDet_SSD'),
                    ('mmdet_fasterrcnn','MMDet_Faster R‑CNN'),
                    ('mmdet_retinanet','MMdet_RetinaNet'), ('mmdet_yolov3', 'MMDet_YOLOv3')
                    ])

MODE_MAPPING = {'O0':'FP32', 'O1':'Mixed', 'fp32':'FP32', 'amp':'AMP'}

def parse_args():
    parser = argparse.ArgumentParser("Generate pt release data.")
    parser.add_argument("--perf_csv", type=str, default='', help="csv file provided by test.")
    parser.add_argument("--metric_file", type=str, default='', help="Input metric json file.")
    parser.add_argument("--device", type=str, default='', help="mlu device type.")
    parser.add_argument("--release_ver", type=str, default='', help="version.")
    parser.add_argument("--pt_ver", type=str, default='', help="version.")
    parser.add_argument("--log_dir", type=str, default='', help="benchmark log for power msg")
    args = parser.parse_args()
    if not args.perf_csv:
        parser.print_usage(sys.stderr)
        sys.exit(-1)
    return args

def iter_all_logfile(opt):
    for filepath, dirnames, filenames in os.walk(opt.log_dir):
        for filename in filenames:
            if filename == "benchmark_log" or (filename.startswith("benchmark_log") and opt.pt_ver in filename):
                yield os.path.join(filepath, filename)

def trans_network_name(net):
    name = net['network name']
    if name in NET_WORK_MAPPING.keys():
        net['network name'] = NET_WORK_MAPPING[name]
    return net

def make_power_dataframe(opt):
    dev = opt.device
    USE_590_DEVICE = False
    USE_370_X8 = False
    if '590' in dev:
        USE_590_DEVICE = True
    elif 'X8' in dev:
        USE_370_X8 = True
    ddp_list = []
    if USE_370_X8:
        ddp_list = [2, 16]
    else:
        ddp_list = [1, 8]
    required_cols = ['net', 'precision', 'cards', 'dura_time']
    more_cards_summary = []
    for i in range (0, 8):
        more_cards_summary.append('card' + str(i) + ' summary')
    precision_list = ['fp32', 'amp']
    if USE_590_DEVICE:
        precision_list.append('tf32')
    power_msg_list = ['min power usage', 'max power usage', 'avg power usage']
    extra_cols = ['net']
    for card in [1, 8]:
        for precision in precision_list:
            for power_msg in power_msg_list:
                extra_cols.append(power_msg + '_' + precision + '_' + str(card))
            extra_cols.append('dura_time_' + precision + '_' + str(card))

    all_bench_log = []
    for bench_log in iter_all_logfile(opt):
        with open(bench_log) as f:
            d = json.load(f)
            if len(d) == 0:
                print(f'[Error]:empty benchmark_log: {bench_log}!')
                exit()
            if isinstance(list(d.values())[0], dict) and "net" in list(d.values())[0]:
                all_bench_log.extend(list(d.values()))
            else:
                all_bench_log.append(d)

    power_msg_dict = dict.fromkeys(extra_cols)
    all_nets_dict_fp32 = dict()
    all_nets_dict_amp = dict()
    all_nets_dict_tf32 = dict()
    net_list_fp32 = []
    net_list_amp = []
    net_list_tf32 = []
    for bench_log in all_bench_log:
        tmp_items = {}
        for k, v in bench_log.items():
            if k in (required_cols + more_cards_summary):
                tmp_items[k] = v
        tmp_power_msg = {}
        for k in tmp_items:
            cards_msg_8 = ((tmp_items['cards'] == 8 and not USE_370_X8) or (tmp_items['cards'] == 16 and USE_370_X8))
            cards_msg_1 = tmp_items['cards'] == 1 or tmp_items['cards'] == 2
            current_key = ''
            if cards_msg_1:
                current_key = k + '_' + tmp_items['precision'].lower() + '_1'
            elif cards_msg_8:
                current_key = k + '_' + tmp_items['precision'].lower() + '_8'
            if k == 'net':
                tmp_power_msg[k] = tmp_items[k]
            if k in more_cards_summary:
                for power_msg in power_msg_list:
                    if cards_msg_1:
                        tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_1'] = tmp_items[k][power_msg]
                    elif cards_msg_8:
                        if power_msg + '_' + tmp_items['precision'] + '_8' not in tmp_power_msg:
                            tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] = tmp_items[k][power_msg]
                        else:
                            tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] += tmp_items[k][power_msg]
            if current_key in extra_cols:
                tmp_power_msg[current_key] = tmp_items[k]

        if tmp_items['cards'] in ddp_list:
            current_precision = tmp_items['precision']
            all_nets_dict = dict()
            net_list = []
            if current_precision.lower() == 'fp32':
                all_nets_dict = all_nets_dict_fp32
                net_list = net_list_fp32
            elif current_precision.lower() == 'tf32':
                all_nets_dict = all_nets_dict_tf32
                net_list = net_list_tf32
            elif current_precision.lower() == 'amp':
                all_nets_dict = all_nets_dict_amp
                net_list = net_list_amp
            if tmp_power_msg['net'] not in net_list:
                all_nets_dict[tmp_power_msg['net']] = tmp_power_msg
                net_list.append(tmp_power_msg['net'])
            else:
                all_nets_dict[tmp_power_msg['net']].update(tmp_power_msg)

    final_df_fp32 = pd.DataFrame(all_nets_dict_fp32)
    final_df_amp = pd.DataFrame(all_nets_dict_amp)
    final_df_tf32 = pd.DataFrame(all_nets_dict_tf32)
    extra_cols_fp32 = ['net']
    extra_cols_amp = ['net']
    extra_cols_tf32 = ['net']
    df_list = []
    extra_cols_list = []
    if USE_590_DEVICE:
        df_list = [final_df_fp32, final_df_amp, final_df_tf32]
        extra_cols_list = [extra_cols_fp32, extra_cols_amp, extra_cols_tf32]
    else:
        df_list = [final_df_fp32, final_df_amp]
        extra_cols_list = [extra_cols_fp32, extra_cols_amp]
    for precision in precision_list:
        current_extra_cols = []
        if precision == 'fp32':
            current_extra_cols = extra_cols_fp32
        elif precision == 'amp':
            current_extra_cols = extra_cols_amp
        elif precision == 'tf32':
            current_extra_cols = extra_cols_tf32
        for card in [1, 8]:
            for power_msg in power_msg_list:
                current_extra_cols.append(power_msg + '_' + precision + '_' + str(card))
            current_extra_cols.append('dura_time_' + precision + '_' + str(card))

    for index, df in enumerate(df_list):
        df_list[index] = df_list[index].T
        df_list[index] = df_list[index][extra_cols_list[index]].copy().reset_index(drop = True)

    return df_list

if __name__ == '__main__':
    opt = parse_args()
    dev = opt.device
    USE_590_DEVICE = False
    if '590' in dev:
        USE_590_DEVICE = True

    precision_df = pd.read_json(opt.metric_file, orient='index')
    metric_df = pd.DataFrame(columns=['network name', 'precision', 'metric_data'])

    for net, record in precision_df.iterrows():
        for k, v in record[dev].items():
            tmp_items = {'network name': net, 'precision': k, 'metric_data': v}
            metric_df = metric_df.append(tmp_items, ignore_index=True)

    origin_perf_df = pd.read_csv(opt.perf_csv)
    required_cols = ['network name', 'dataset', 'precision', 'batch_size']

    if dev in ["MLU370-M8", "MLU590-H8", "MLU590-M9", "MLU590-M9U"]:
        required_cols += ['card1 throughput', 'card8 throughput', 'card8 ddp ratio']
    elif dev in ["MLU370-X4"]:
        required_cols += ['card1 throughput', 'card4 throughput',  'card4 ddp ratio', 'card8 throughput', 'card8 ddp ratio']
    elif dev in ["MLU370-X8"]:
        required_cols += ['card2 throughput', 'card8 throughput', 'card4 ddp ratio', 'card16 throughput', 'card8 ddp ratio']
    else:
        print(f'[Error]: invaild device: {dev}')
        exit(1)

    precision_list = ['fp32', 'amp']
    if USE_590_DEVICE:
        precision_list.append('tf32')

    power_msg_list = ['min power usage', 'max power usage', 'avg power usage', 'dura_time']

    origin_perf_df['card4 ddp ratio']='N/A'
    origin_perf_df['card8 ddp ratio']='N/A'
    perf_df = origin_perf_df[required_cols].copy()
    power_df_fp32 = None
    power_df_amp = None
    power_df_tf32 = None
    perf_df_fp32 = None
    perf_df_amp = None
    perf_df_tf32 = None
    perf_df_list  = []
    power_cols_list = []
    for precision in precision_list:
        power_cols = []
        for card in [1, 8]:
            for power_msg in power_msg_list:
                power_cols.append(power_msg + '_' + precision + '_' + str(card))
        power_cols_list.append(power_cols)

    if USE_590_DEVICE:
        power_df_fp32, power_df_amp, power_df_tf32 = make_power_dataframe(opt)
        perf_df_fp32 = perf_df.loc[perf_df['precision'] == 'fp32'].copy()
        perf_df_amp = perf_df.loc[perf_df['precision'] == 'amp'].copy()
        perf_df_tf32 = perf_df.loc[perf_df['precision'] == 'tf32'].copy()
        perf_df_fp32 = pd.merge(perf_df_fp32, power_df_fp32, left_on = 'network name', right_on = 'net', how='left')
        perf_df_amp = pd.merge(perf_df_amp, power_df_amp, left_on = 'network name', right_on = 'net', how='left')
        perf_df_tf32 = pd.merge(perf_df_tf32, power_df_tf32, left_on = 'network name', right_on = 'net', how='left')
        perf_df_list  = [perf_df_fp32, perf_df_amp, perf_df_tf32]
    else:
        power_df_fp32, power_df_amp = make_power_dataframe(opt)
        perf_df_fp32 = perf_df.loc[perf_df['precision'] == 'fp32'].copy()
        perf_df_amp = perf_df.loc[perf_df['precision'] == 'amp'].copy()
        perf_df_fp32 = pd.merge(perf_df_fp32, power_df_fp32, left_on = 'network name', right_on = 'net', how='left')
        perf_df_amp = pd.merge(perf_df_amp, power_df_amp, left_on = 'network name', right_on = 'net', how='left')
        perf_df_list  = [perf_df_fp32, perf_df_amp]

    final_df = []
    for i, df in enumerate(perf_df_list):
        for index, row in perf_df_list[i].iterrows():
            net = row['network name']
            metric = row['precision']
            if net in NET_WORK_MAPPING.keys():
                perf_df_list[i].loc[index, 'network name'] = NET_WORK_MAPPING[net]
            if metric in MODE_MAPPING.keys():
                perf_df_list[i].loc[index, 'precision'] = MODE_MAPPING[metric]

        df_new = perf_df_list[i].merge(metric_df,  on=['network name', 'precision'], how = 'left')
        if dev in ["MLU370-M8", "MLU590-H8", "MLU590-M9", "MLU590-M9U"]:
            df_new['card8 ddp ratio'] = df_new.apply(lambda x : str(round(pd.to_numeric(x['card8 throughput'], errors='coerce') / pd.to_numeric(x['card1 throughput'], errors='coerce')/ 8 * 100, 2)) + '%', axis = 1)
        elif dev in ["MLU370-X4"]:
            df_new['card4 ddp ratio'] = df_new.apply(lambda x : str(round(pd.to_numeric(x['card4 throughput'], errors='coerce') / pd.to_numeric(x['card1 throughput'], errors='coerce')/ 4 * 100, 2)) + '%', axis = 1)
            df_new['card8 ddp ratio'] = df_new.apply(lambda x : str(round(pd.to_numeric(x['card8 throughput'], errors='coerce') / pd.to_numeric(x['card1 throughput'], errors='coerce')/ 8 * 100, 2)) + '%', axis = 1)
        elif dev in ["MLU370-X8"]:
            df_new['card4 ddp ratio'] = df_new.apply(lambda x : str(round(pd.to_numeric(x['card8 throughput'], errors='coerce') / pd.to_numeric(x['card2 throughput'], errors='coerce')/ 4 * 100, 2)) + '%', axis = 1)
            df_new['card8 ddp ratio'] = df_new.apply(lambda x : str(round(pd.to_numeric(x['card16 throughput'], errors='coerce') / pd.to_numeric(x['card2 throughput'], errors='coerce')/ 8 * 100, 2)) + '%', axis = 1)

        select_cols = required_cols + ['metric_data'] + power_cols_list[i]
        df_gen = df_new[select_cols]
        df_gen = df_gen.rename(columns={'network name':'网络名称', 'dataset':'数据集', 'metric_data':'精度',
            'precision':'模式', 'card1 throughput':'单卡','card4 throughput':'四卡', 'unit':'单位', 'card2 throughput':'单整卡','card8 throughput':'四整卡' if dev == "MLU370-X8" else '八卡',
            'card16 throughput':'八整卡', 'card4 ddp ratio':'四卡扩展率', 'card8 ddp ratio':'八卡扩展率',
            'min power usage_fp32_1':'单卡最小功率', 'max power usage_fp32_1':'单卡最大功率', 'avg power usage_fp32_1':'单卡平均功率', 'dura_time_fp32_1':'单卡运行时间',
            'min power usage_fp32_8':'八卡最小功率', 'max power usage_fp32_8':'八卡最大功率', 'avg power usage_fp32_8':'八卡平均功率', 'dura_time_fp32_8':'八卡运行时间',
            'min power usage_amp_1':'单卡最小功率', 'max power usage_amp_1':'单卡最大功率', 'avg power usage_amp_1':'单卡平均功率', 'dura_time_amp_1':'单卡运行时间',
            'min power usage_amp_8':'八卡最小功率', 'max power usage_amp_8':'八卡最大功率', 'avg power usage_amp_8':'八卡平均功率', 'dura_time_amp_8':'八卡运行时间',
            'min power usage_tf32_1':'单卡最小功率', 'max power usage_tf32_1':'单卡最大功率', 'avg power usage_tf32_1':'单卡平均功率', 'dura_time_tf32_1':'单卡运行时间',
            'min power usage_tf32_8':'八卡最小功率', 'max power usage_tf32_8':'八卡最大功率', 'avg power usage_tf32_8':'八卡平均功率', 'dura_time_tf32_8':'八卡运行时间'
            })
        df_gen.fillna('N/A', inplace=True)
        final_df.append(df_gen)

    file_name = 'Cambricon-PyTorch-Performance-'+ '-' + opt.release_ver + '-' + opt.pt_ver + '-' + dev + '.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for index, df in enumerate(final_df):
        df.to_excel(writer, sheet_name=precision_list[index].upper())
    writer.save()

