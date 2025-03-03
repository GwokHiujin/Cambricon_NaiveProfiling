import os
import json
import sys
import argparse
import copy
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def parse_args():
    parser = argparse.ArgumentParser("gen release tpi.")
    parser.add_argument("--log_dir", type=str, default='', help="daily benchmark log directory.")
    parser.add_argument("--prev_log_dir", type=str, default='', help="previously release benchmark log directory.")
    parser.add_argument("--perf_csv", type=str, default='', help="daily performance csv file.") # reserve for time_to_train and keep compatablility
    parser.add_argument("--device", type=str, default='', help="mlu device.")
    parser.add_argument("--release_ver", type=str, default='', help="release version.")
    parser.add_argument("--prev_ver", type=str, default='', help="previously release version.")
    parser.add_argument("--simplified", action='store_true',default=False, help="generate verbose tpi or not.")
    parser.add_argument("--pt_ver", type=str, default='', help="pytorch version.")

    args = parser.parse_args()
    if not args.log_dir:
        parser.print_usage(sys.stderr)
        sys.exit(-1)
    return args
opt = parse_args()

def iter_all_logfile(opt):
    for filepath, dirnames, filenames in os.walk(opt.log_dir):
        for filename in filenames:
            if filename == "benchmark_log" or (filename.startswith("benchmark_log") and opt.pt_ver in filename):
                yield os.path.join(filepath, filename)

def make_csv(opt):
    dev = opt.device
    ddp_list = []
    if dev in ["MLU370-M8", "MLU370-X4", "MLU590-H8", "MLU590-M9", "MLU590-M9U"]:
        ddp_list = [1,8]
    elif dev in ["MLU370-X8"]:
        ddp_list = [2,16]
    else:
        print(f'[Error]:Unknown device: {dev}!')
        exit()
    USE_590_DEVICE = False
    USE_370_X8 = False
    if "590" in dev:
        USE_590_DEVICE = True
    elif "X8" in dev:
        USE_370_X8 = True

    # these cols are keys in benchmark log
    required_cols = ['net', 'batch_size', 'throughput',  'device', 'batch_time_avg', 'hardware_time_avg', 'cards', 'precision', 'dura_time']
    more_cards_summary = []
    for i in range (0, 8):
        more_cards_summary.append('card' + str(i) + ' summary')

    precision_list = ['fp32', 'amp']
    if USE_590_DEVICE:
        precision_list.append('tf32')
    # these cols will multiply with precision
    level2_msg_list = ['min power usage', 'max power usage', 'avg power usage']
    item_list = ['throughput', 'hardware_time_avg', 'batch_time_avg']

    # these cols are common to all devices
    extra_cols = ['net', 'batch_size', 'time_to_train']

    # multiply precision with the above item
    for card in [1, 8]:
        for precision in precision_list:
            for item in item_list:
                extra_cols.append(item + '_' + precision + '_' + str(card))
            for power_msg in level2_msg_list:
                extra_cols.append(power_msg + '_' + precision + '_' + str(card))
            extra_cols.append('dura_time_' + precision + '_' + str(card))

    calculated_item_list = ['ddp_speedup_ratio_fp32', 'ddp_speedup_ratio_amp', 'hwtime_e2etime_ratio_fp32', 'hwtime_e2etime_ratio_amp', 'mixed_throughput_leverage']
    if USE_590_DEVICE:
        calculated_item_list.append('ddp_speedup_ratio_tf32')
        calculated_item_list.append('tf32_throughput_leverage')
        calculated_item_list.append('hwtime_e2etime_ratio_tf32')
    perf_df = pd.DataFrame(columns = extra_cols + calculated_item_list)

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

    all_msg_dict = dict.fromkeys(extra_cols)
    all_nets_dict = dict()
    net_list = []
    for bench_log in all_bench_log:
        tmp_items = {}
        for k, v in bench_log.items():
            if k in (required_cols + more_cards_summary):
                tmp_items[k] = v
        tmp_msg = {}
        for k in tmp_items:
            cards_msg_8 = ((tmp_items['cards'] == 8 and not USE_370_X8) or (tmp_items['cards'] == 16 and USE_370_X8))
            cards_msg_1 = tmp_items['cards'] == 1 or tmp_items['cards'] == 2
            current_key = ''
            if cards_msg_1:
                current_key = k + '_' + tmp_items['precision'].lower() + '_1'
            elif cards_msg_8:
                current_key = k + '_' + tmp_items['precision'].lower() + '_8'
            # deal with cols without precision and cards
            if k == 'net':
                tmp_msg[k] = tmp_items[k]
            if k == 'batch_size':
                tmp_msg[k] = tmp_items[k]
            # currently this item is empty, it may be provided by the csv mentioned in the argparse part
            # in the future
            tmp_msg['time_to_train'] = ''
            if k in more_cards_summary:
                for power_msg in level2_msg_list:
                    if cards_msg_1:
                        tmp_msg[power_msg + '_' + tmp_items['precision'].lower() + '_1'] = tmp_items[k][power_msg]
                    elif cards_msg_8:
                        if power_msg + '_' + tmp_items['precision'] + '_8' not in tmp_msg:
                            tmp_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] = tmp_items[k][power_msg]
                        else:
                            tmp_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] += tmp_items[k][power_msg]
            if current_key in extra_cols:
                tmp_msg[current_key] = tmp_items[k]
        if tmp_items['cards'] in ddp_list:
            if tmp_msg['net'] not in net_list:
                all_nets_dict[tmp_msg['net']] = tmp_msg
                net_list.append(tmp_msg['net'])
            else:
                all_nets_dict[tmp_msg['net']].update(tmp_msg)

    for net in all_nets_dict:
        for calculated_item in calculated_item_list:
            # python 3.10 has match key word, but I cant assume that is avaliable
            current_net = all_nets_dict[net]
            if calculated_item == 'ddp_speedup_ratio_tf32':
                current_net[calculated_item] = current_net.get('throughput_tf32_8', np.NaN) / current_net.get('throughput_tf32_1', np.NaN) / 8 * 100
            elif calculated_item == 'ddp_speedup_ratio_amp':
                current_net[calculated_item] = current_net.get('throughput_amp_8', np.NaN) / current_net.get('throughput_amp_1', np.NaN) / 8 * 100
            elif calculated_item == 'ddp_speedup_ratio_fp32':
                current_net[calculated_item] = current_net.get('throughput_fp32_8', np.NaN) / current_net.get('throughput_fp32_1', np.NaN) / 8 * 100
            elif calculated_item == 'hwtime_e2etime_ratio_fp32':
                current_net[calculated_item] = current_net.get('hardware_time_avg_fp32_1', np.NaN) / current_net.get('batch_time_avg_fp32_1', np.NaN) * 100
            elif calculated_item == 'hwtime_e2etime_ratio_tf32':
                current_net[calculated_item] = current_net.get('hardware_time_avg_tf32_1', np.NaN) / current_net.get('batch_time_avg_tf32_1', np.NaN) * 100
            elif calculated_item == 'hwtime_e2etime_ratio_amp':
                current_net[calculated_item] = current_net.get('hardware_time_avg_amp_1', np.NaN) / current_net.get('batch_time_avg_amp_1', np.NaN) * 100
            elif calculated_item == 'mixed_throughput_leverage':
                current_net[calculated_item] = current_net.get('throughput_amp_1', np.NaN) / current_net.get('throughput_fp32_1', np.NaN) * 100
            elif calculated_item == 'tf32_throughput_leverage':
                current_net[calculated_item] = current_net.get('throughput_tf32_1', np.NaN) / current_net.get('throughput_fp32_1', np.NaN) * 100

    perf_df = pd.DataFrame(all_nets_dict)
    perf_df = perf_df.T
    final_df = perf_df[extra_cols + calculated_item_list].copy().reset_index(drop = True)

    if opt.simplified:
        return final_df
    else:
        final_df.to_csv('PyTorch-TPI'+ '-' + opt.release_ver + '-' + opt.pt_ver + '-' + dev + '.csv')

def make_simplified_csv(opt):
    opt_release = copy.deepcopy(opt)
    opt_prev = copy.deepcopy(opt)
    tpi_release = make_csv(opt_release)
    opt_prev.release_ver = opt.prev_ver
    opt_prev.log_dir = opt.prev_log_dir
    tpi_prev = make_csv(opt_prev)
    USE_590_DEVICE = False
    if '590' in opt.device:
        USE_590_DEVICE = True
    # well, select cols by hands
    select_cols_fp32 = ['net', 'batch_size', 'time_to_train', 'throughput_fp32_1', 'throughput_fp32_8', 'ddp_speedup_ratio_fp32', 'hwtime_e2etime_ratio_fp32', 'hardware_time_avg_fp32_1', 'min power usage_fp32_1', 'max power usage_fp32_1', 'avg power usage_fp32_1', 'dura_time_fp32_1', 'min power usage_fp32_8', 'max power usage_fp32_8', 'avg power usage_fp32_8', 'dura_time_fp32_8']
    select_cols_amp = ['net', 'batch_size', 'time_to_train', 'throughput_amp_1', 'throughput_amp_8', 'ddp_speedup_ratio_amp',  'mixed_throughput_leverage', 'hwtime_e2etime_ratio_amp', 'hardware_time_avg_amp_1', 'min power usage_amp_1', 'max power usage_amp_1', 'avg power usage_amp_1', 'dura_time_amp_1', 'min power usage_amp_8', 'max power usage_amp_8', 'avg power usage_amp_8', 'dura_time_amp_8']
    select_cols_tf32 = []
    if USE_590_DEVICE:
        select_cols_tf32 = ['net', 'batch_size', 'time_to_train', 'throughput_tf32_1', 'throughput_tf32_8', 'ddp_speedup_ratio_tf32',  'tf32_throughput_leverage', 'hwtime_e2etime_ratio_tf32', 'hardware_time_avg_tf32_1', 'min power usage_tf32_1', 'max power usage_tf32_1', 'avg power usage_tf32_1', 'dura_time_tf32_1', 'min power usage_tf32_8', 'max power usage_tf32_8', 'avg power usage_tf32_8', 'dura_time_tf32_8']
    tpi_release_fp32 = tpi_release[select_cols_fp32].copy()
    tpi_release_amp = tpi_release[select_cols_amp].copy()
    if USE_590_DEVICE:
        tpi_release_tf32 = tpi_release[select_cols_tf32].copy()
    tpi_prev_fp32 = tpi_prev[select_cols_fp32].copy()
    tpi_prev_amp = tpi_prev[select_cols_amp].copy()
    if USE_590_DEVICE:
        tpi_prev_tf32 = tpi_prev[select_cols_tf32].copy()
    required_cols = ['net', 'batch_size']

    tpi_simplified_fp32 = pd.DataFrame(columns = required_cols)
    tpi_simplified_amp = pd.DataFrame(columns = required_cols)
    tpi_simplified_tf32 = None
    if USE_590_DEVICE:
        tpi_simplified_tf32 = pd.DataFrame(columns = required_cols)
    tpi_list = [tpi_simplified_fp32, tpi_simplified_amp]
    tpi_release_list = [tpi_release_fp32, tpi_release_amp]
    tpi_prev_list = [tpi_prev_fp32, tpi_prev_amp]
    cols_list = [select_cols_fp32, select_cols_amp]
    if USE_590_DEVICE:
        tpi_list.append(tpi_simplified_tf32)
        cols_list.append(select_cols_tf32)
        tpi_release_list.append(tpi_release_tf32)
        tpi_prev_list.append(tpi_prev_tf32)

    for index, tpi_simplified in enumerate(tpi_list):
        i = 3
        for k in cols_list[index]:
            if k == 'net':
                tpi_list[index]['net'] = tpi_release_list[index][k]
            elif k == 'batch_size':
                tpi_list[index]['batch_size'] = tpi_release_list[index][k]
            elif k == 'time_to_train':
                tpi_list[index]['time_to_train'] = tpi_release_list[index][k]
            elif ('power' in k or 'dura' in k):
                tpi_list[index][k] = tpi_release_list[index][k]
            else:
                tpi_list[index] = pd.merge(tpi_list[index], tpi_release_list[index][['net', k]], on='net', how='left')
                tpi_list[index] = tpi_list[index].rename(columns={k:opt.release_ver})
                tpi_list[index] = pd.merge(tpi_list[index], tpi_prev_list[index][['net', k]], on='net', how='left')
                tpi_list[index] = tpi_list[index].rename(columns={k:opt.prev_ver})
                if is_numeric_dtype(tpi_release_list[index][k]):
                    tpi_list[index].insert(loc = i + 2, column='diff(%)', value = (tpi_list[index].iloc[:, i] - tpi_list[index].iloc[:, i + 1]) / tpi_list[index].iloc[:, i + 1], allow_duplicates=True)
                else:
                    temp_release = tpi_list[index].iloc[:, i].astype(str).str.replace('%', '')
                    temp_release = pd.to_numeric(temp_release, errors='coerce')
                    temp_prev = tpi_list[index].iloc[:, i + 1].astype(str).str.replace('%', '')
                    temp_prev = pd.to_numeric(temp_prev, errors='coerce')
                    tpi_list[index].insert(loc = i + 2, column='diff(%)', value = (temp_release - temp_prev) / temp_prev, allow_duplicates=True)
                i = i + 3

    if USE_590_DEVICE:
        tpi_simplified_fp32, tpi_simplified_amp, tpi_simplified_tf32 = tpi_list
    else:
        tpi_simplified_fp32, tpi_simplified_amp = tpi_list

    precision_list = ['FP32', 'mixed precision']
    if USE_590_DEVICE:
        precision_list.append('TF32')
    for index, tpi_list[index] in enumerate(tpi_list):
        current_precision = precision_list[index]
        tpi_list[index]['diff(%)'] = tpi_list[index]['diff(%)'].applymap(lambda x: '{:.2%}'.format(x))
        tpi_list[index] = tpi_list[index].rename(columns={'net':''})
        tpi_list[index] = tpi_list[index].rename(columns={'batch_size':''})
        tpi_list[index] = tpi_list[index].rename(columns={'time_to_train':''})
        old_columns = tpi_list[index].columns
        new_columns = ['Model', 'max batch size', 'time_to_train']
        candidate_columns = ['Throughput-' + current_precision, 'Throughput-' + current_precision + 'Card 8', 'Weak scaling-' + current_precision + 'Card 8']
        if current_precision == 'mixed precision':
            candidate_columns.append('mixed precision Throughput Leverage(%)')
        elif current_precision == 'TF32':
            candidate_columns.append('TF32 Throughput Leverage(%)')
        candidate_columns.append('HW time / E2E time ratio(%)')
        candidate_columns.append('HW time')
        candidate_columns = np.repeat(candidate_columns, 3)
        candidate_columns = candidate_columns.tolist()
        power_columns = ['Single card power usage', '8 cards power usage']
        power_columns = np.repeat(power_columns, 4)
        power_columns = power_columns.tolist()
        new_columns = new_columns + candidate_columns + power_columns
        tpi_list[index].columns = [new_columns, old_columns]

    if USE_590_DEVICE:
        tpi_simplified_fp32, tpi_simplified_amp, tpi_simplified_tf32 = tpi_list
    else:
        tpi_simplified_fp32, tpi_simplified_amp = tpi_list

    file_name = 'PyTorch-TPI'+ '-' + opt.release_ver + '-' + opt.pt_ver + '-' + opt.device + '.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    if USE_590_DEVICE:
        tpi_simplified_fp32.to_excel(writer, sheet_name='FP32')
        tpi_simplified_amp.to_excel(writer, sheet_name='AMP')
        tpi_simplified_tf32.to_excel(writer, sheet_name='TF32')
        writer.sheets['FP32'].set_row(2, None, None, {'hidden': True})
        writer.sheets['AMP'].set_row(2, None, None, {'hidden': True})
        writer.sheets['TF32'].set_row(2, None, None, {'hidden': True})
    else:
        tpi_simplified_fp32.to_excel(writer, sheet_name='FP32')
        tpi_simplified_amp.to_excel(writer, sheet_name='AMP')
        writer.sheets['FP32'].set_row(2, None, None, {'hidden': True})
        writer.sheets['AMP'].set_row(2, None, None, {'hidden': True})
    writer.save()

if __name__ == "__main__":
    if opt.simplified:
        make_simplified_csv(opt)
    else:
        make_csv(opt)
