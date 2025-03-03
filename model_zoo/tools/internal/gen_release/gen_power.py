import os
import json
import sys
import argparse
import copy
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("gen release tpi.")
    parser.add_argument("--log_dir", type=str, default='', help="daily benchmark log directory.")
    parser.add_argument("--device", type=str, default='', help="mlu device.")
    parser.add_argument("--release_ver", type=str, default='', help="release version.")
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
            if filename == "benchmark_log":
                yield os.path.join(filepath, filename)

def make_csv(opt):
    dev = opt.device
    ddp_list = [1, 8]
    required_cols = ['net', 'precision', 'cards', 'dura_time']
    perf_cols = ['net']
    more_cads_summary = []
    for i in range (0, 8):
        more_cads_summary.append('card' + str(i) + ' summary')
    precision_list = ['fp32', 'tf32', 'amp']
    power_msg_list = ['min power usage', 'max power usage', 'avg power usage']
    extra_cols = []
    for card in ddp_list:
        for precision in precision_list:
            for power_msg in power_msg_list:
                extra_cols.append(power_msg + '_' + precision + '_' + str(card))
            extra_cols.append('dura_time_' + precision + '_' + str(card))
    perf_df = pd.DataFrame(columns = perf_cols + extra_cols)

    all_bench_log = []
    for bench_log in iter_all_logfile(opt):
        with open(bench_log) as f:
            d = json.load(f)
            all_bench_log.append(d)

    power_msg_dict = dict.fromkeys(perf_cols + extra_cols)
    all_nets_dict = dict()
    net_list = []
    for bench_log in all_bench_log:
        tmp_items = {}
        for k, v in bench_log.items():
            if k in (required_cols + more_cads_summary):
                tmp_items[k] = v
        tmp_power_msg = {}
        for k in tmp_items:
            if k == 'net':
                tmp_power_msg[k] = tmp_items[k]
            if k in more_cads_summary:
                for power_msg in power_msg_list:
                    if tmp_items['cards'] == 1:
                        tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_1'] = tmp_items[k][power_msg]
                    elif tmp_items['cards'] == 8:
                        if power_msg + '_' + tmp_items['precision'] + '_8' not in tmp_power_msg:
                            tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] = tmp_items[k][power_msg]
                        else:
                            tmp_power_msg[power_msg + '_' + tmp_items['precision'].lower() + '_8'] += tmp_items[k][power_msg]
            if k == 'dura_time' and tmp_items['cards'] in ddp_list:
                tmp_power_msg['dura_time_' + tmp_items['precision'].lower() + '_' + str(tmp_items['cards'])] = tmp_items[k]

        if tmp_items['cards'] in ddp_list:
            if tmp_power_msg['net'] not in net_list:
                all_nets_dict[tmp_power_msg['net']] = tmp_power_msg
                net_list.append(tmp_power_msg['net'])
            else:
                all_nets_dict[tmp_power_msg['net']].update(tmp_power_msg)

    perf_df = pd.DataFrame(all_nets_dict)
    perf_df = perf_df.T
    final_df = perf_df[perf_cols + extra_cols].copy().reset_index(drop = True)
    final_df = final_df.rename(columns = {'net':''})
    final_df = final_df.rename(columns = {'min power usage_fp32_1':'min', 'max power usage_fp32_1':'max', 'avg power usage_fp32_1':'avg', 'dura_time_fp32_1':'dura_time'})
    final_df = final_df.rename(columns = {'min power usage_tf32_1':'min', 'max power usage_tf32_1':'max', 'avg power usage_tf32_1':'avg', 'dura_time_tf32_1':'dura_time'})
    final_df = final_df.rename(columns = {'min power usage_amp_1':'min', 'max power usage_amp_1':'max', 'avg power usage_amp_1':'avg', 'dura_time_amp_1':'dura_time'})
    final_df = final_df.rename(columns = {'min power usage_fp32_8':'min', 'max power usage_fp32_8':'max', 'avg power usage_fp32_8':'avg', 'dura_time_fp32_8':'dura_time'})
    final_df = final_df.rename(columns = {'min power usage_tf32_8':'min', 'max power usage_tf32_8':'max', 'avg power usage_tf32_8':'avg', 'dura_time_tf32_8':'dura_time'})
    final_df = final_df.rename(columns = {'min power usage_amp_8':'min', 'max power usage_amp_8':'max', 'avg power usage_amp_8':'avg', 'dura_time_amp_8':'dura_time'})

    old_columns = final_df.columns
    new_columns = ['Model']
    candidate_columns = ['FP32 power usage', 'TF32 power usage', 'mixed precision power usage', 'FP32 power usage - Card8', 'TF32 power usage - Card8', 'mixed precision power usage - Card8']
    candidate_columns = np.repeat(candidate_columns, 4)
    candidate_columns = candidate_columns.tolist()
    new_columns = new_columns + candidate_columns
    final_df.columns = [new_columns, old_columns]
    def get_header(precision):
        result = [('Model', '')]
        postfix_list = [' power usage', ' power usage - Card8']
        second_level_header = ['min', 'max', 'avg', 'dura_time']
        for postfix in postfix_list:
            for item in second_level_header:
                result.append(tuple((precision + postfix, item)))
        return result
    final_fp32 = final_df.loc[:, final_df.columns.isin(get_header('FP32'))].copy()
    final_tf32 = final_df.loc[:, final_df.columns.isin(get_header('TF32'))].copy()
    final_amp = final_df.loc[:, final_df.columns.isin(get_header('mixed precision'))].copy()
    file_name = 'PyTorch-Power' + '-' + opt.release_ver + '-' + opt.pt_ver + '-' + dev + '.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    final_fp32.to_excel(writer, sheet_name='FP32')
    final_tf32.to_excel(writer, sheet_name='TF32')
    final_amp.to_excel(writer, sheet_name='mixed precision')
    writer.sheets['FP32'].set_row(2, None, None, {'hidden': True})
    writer.sheets['TF32'].set_row(2, None, None, {'hidden': True})
    writer.sheets['mixed precision'].set_row(2, None, None, {'hidden': True})
    writer.save()

if __name__ == "__main__":
    make_csv(opt)
