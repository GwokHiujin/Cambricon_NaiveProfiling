## 1、介绍
本工具用于统计模型中运行的所有信息，并上传到superset网页展示。

***模型的信息包括：***
- 软件信息：pytorch框架及其依赖库的版本信息
- 硬件信息：mlu型号，cpu型号
- 模型超参：batch_size，opt_level，extra_params
- 性能数据：latency_stats，throughput

***superset网页链接***
> http://dataview.cambricon.com/superset/dashboard/18/

> http://dataview.cambricon.com/superset/dashboard/14/

## 2、运行步骤
### 2.1 环境配置
> pip install git+http://gitlab.software.cambricon.com/liangfan/cndb

### 2.2 准备性能测试数据
- 在tools/internal/superset目录下新建input目录，将性能测试结果benchmark_log文件的压缩包拷入input目录，然后执行解压命令
    > for f in *.tar.gz;do tar xvzf "$f" -C ./;done;rm *.tar.gz
- 更新tools/internal/superset/soft_info.json中维护的catch和CTR的版本信息。

### 2.3 运行命令

启动展示性能数据的脚本

> bash launch.sh 0

启动展示精度数据的脚本

> bash launch.sh 1

