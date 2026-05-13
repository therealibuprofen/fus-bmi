# py_fus_decoder

`py_fus_decoder` 是一个面向功能性超声（fUS）解码任务的科研级 Python 评估框架，参考本仓库现有 MATLAB `decoders/` 的统一训练/推理/交叉验证思路，补齐多数据集兼容、传统模型基线、深度学习模型和离线统一评测入口。

## 目标

- 兼容不同物种、不同下游任务的数据集
- 支持统一的数据规范化与数据集适配
- 提供 baseline 解码器
  - `pca_lda`
  - `cpca_lda`
- 提供深度学习解码器
  - `cnn`
  - `cnn_lstm`
- 提供离线统一评测脚本
  - 公平交叉验证
  - 基线对比
  - 模型能力分析
  - 汇总 JSON 输出

## 目录

```text
py_fus_decoder/
├── README.md
├── pyproject.toml
├── configs/
│   └── example_offline_eval.json
├── scripts/
│   └── offline_benchmark.py
└── fus_decoder/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── utils.py
    ├── data/
    │   ├── __init__.py
    │   ├── adapters.py
    │   └── base.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── analysis.py
    │   ├── cv.py
    │   ├── metrics.py
    │   └── runner.py
    └── models/
        ├── __init__.py
        ├── base.py
        ├── classical.py
        ├── deep.py
        └── registry.py
```

## 数据格式

框架将不同来源的数据集统一为如下规范：

- `samples`
  - 体数据：`[N, H, W, T]`
  - 扁平特征：`[N, F]`
- `labels`
  - `N` 个样本对应的类别标签
- `metadata`
  - 可选，支持 `species`、`task`、`session`、`subject_id`、`groups` 等字段

目前支持的输入方式：

- `.npz`
- `.mat`
- 目录型数据集
  - `samples.npy`
  - `labels.npy`
  - `metadata.json`

字段名支持自动识别，例如：

- `data` / `labels`
- `X` / `y`
- `features` / `labels`
- `train_data` / `train_labels`
- `samples` / `labels`

## 安装

如果需要完整功能，建议安装可选依赖：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder
pip install -e ".[full]"
```

如果只先看代码结构，不安装依赖也可以；运行时缺少依赖会给出明确报错。

## 运行示例

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/offline_benchmark.py \
  --config py_fus_decoder/configs/example_offline_eval.json
```

数据量实验示例：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/offline_benchmark.py \
  --config py_fus_decoder/configs/data_regime_binary_experiment.json
```

输出内容包括：

- 每个模型的折级别指标
- 汇总统计
- 混淆矩阵
- 模型参数摘要
- `species/task/session` 等元数据分层分析
- 不同训练集比例（如 `10/30/50/100%`）下的方法对比

## 配置说明

见 [example_offline_eval.json](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/configs/example_offline_eval.json)。

## 设计说明

- 传统模型使用 sklearn 风格接口
- 深度模型使用 PyTorch 风格训练器
- 评测入口统一通过 `OfflineEvaluationRunner`
- 数据适配层负责把不同实验物种/任务的数据映射到统一 schema

当前 baseline 定义与本项目需求一致：

- `PCA + LDA`
  - 先用 PCA 保留 `95%` 数据方差
  - 再用 LDA 提升类别可分性
- `cPCA + LDA`
  - 先用 class-wise PCA 建立类别相关子空间
  - 每类子空间默认同样保留 `95%` 方差
  - 再用 LDA 做判别

当前深度模型默认按小模型设计：

- `CNN`
  - 默认 `2` 层卷积
  - 默认小卷积核 `3x3x3`
  - 默认少通道 `4, 8`
  - 默认小全连接隐藏层 `32`
- `CNN+LSTM`
  - 默认 `2` 层 2D CNN 编码
  - 默认少通道 `4, 8`
  - 默认 `1` 层 LSTM
  - 默认 hidden size `32`

这让后续继续扩展：

- 新物种数据适配器
- 新任务标签体系
- 新模型结构
- 新指标与可解释性分析

都只需要修改局部模块
