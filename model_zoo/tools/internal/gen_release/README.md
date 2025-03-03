
**requirements:**
- pandas == 1.4.4
- json
- numpy
- xlsxwriter

generate performance sheet:

```shell
    python gen_release_data.py --perf_csv ../v16/MLU370-X4_perf_test_202241129_1.9.csv --metric_file ../data/precision_torch1.9.json --device MLU370-X4 --release_ver v1.2.1 --pt_ver torch1.9 --log_dir ./370x8
```
- `--metric_file {}`: metric json file.
- `--log_dir {BENCHMARK_LOG}`: released benchmark log for power usage.

generate tpi sheet:

```shell
    python gen_tpi.py --log_dir ./370x8r1.9_1.6.0 --prev_log_dir ./370x8_1.3.2 --device MLU370-X8 --prev_ver v1.3.2 --release_ver v1.6.0 --pt_ver torch1.9 --perf_csv ./MLU370X8_r1.9_perf_test_2022971714.csv --simplified
```
- `--log_dir {BENCHMARK_LOG}` released benchmark log.
- `--prev_log_dir {BENCHMARK_LOG}` previously released benchmark log.
- `--perf_csv`: daily perf csv file.
- `--device {DEVICE}`: `MLU290-M5`, `MLU370-X4`,etc
- `--release_ver {RELEASE_VERSION}`:
- `--prev_ver {PREV_RELEASE_VERSION}`:
- `--pt_ver {PYTORCH_VERSION}`:
- `--simplified` produce simplified tpi file.

generate power sheet:

```shell
    python gen_power.py --log_dir ./590h8pt1.6 --device MLU590-H8 --release_ver v1.2.1 --pt_ver torch1.6
```
- `--log_dir {BENCHMARK_LOG}` released benchmark log.
- `--device {DEVICE}`: `MLU590-H8`, `MLU590-M9`
- `--release_ver {RELEASE_VERSION}`:
- `--pt_ver {PYTORCH_VERSION}`:

