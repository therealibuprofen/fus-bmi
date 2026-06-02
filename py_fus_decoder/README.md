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

最简单的方式是在编辑器里直接打开下面任意文件，然后点击运行按钮：

- [run_cross_session.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/run_cross_session.py:1)
- [run_group_sessions.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/run_group_sessions.py:1)
- [run_offline_smoke.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/run_offline_smoke.py:1)
- [run_data_regime_binary.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/run_data_regime_binary.py:1)
- [run_data_regime_8target.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/run_data_regime_8target.py:1)

如果使用 VS Code 或 Cursor，也可以直接在 Run and Debug 里选择：

- `fUS: Cross Session LOSO`
- `fUS: Group Sessions`
- `fUS: Offline Smoke`
- `fUS: Data Regime Binary`
- `fUS: Data Regime 8 Target`

命令行也可以简化成零参数运行：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/run_cross_session.py
python3 py_fus_decoder/run_group_sessions.py
python3 py_fus_decoder/run_offline_smoke.py
```

`offline_benchmark.py` 默认会通过 GUI 交互选择数据集路径：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/offline_benchmark.py
```

如果希望完全使用配置文件里的路径，不弹出 GUI：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/offline_benchmark.py --no-gui-select-dataset
```

数据量实验示例：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/run_data_regime_binary.py
python3 py_fus_decoder/run_data_regime_8target.py
```

多 session 条件分组示例：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/group_sessions_by_condition.py
```

cross-session leave-one-session-out 评测示例：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/cross_session_leave_one_session_out.py
```

默认会弹出交互窗口：

- 先选择 `3` 个或更多 `.mat` session 文件，或选择一个包含 `.mat` 的 dataset 文件夹
- 脚本自动根据 `project_record.json` 读取 `Monkey/Slot/Task/nTargets`
- 自动按 `Monkey + Slot + Task + nTargets` 分组
- 如果存在多个可用组，会弹窗让你选择一个组
- 如果组内超过 `3` 个 session，会弹窗让你选择参与 LOSO 的 session

如果希望完全使用配置文件，不弹出交互窗口：

```bash
cd /Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI
python3 py_fus_decoder/scripts/cross_session_leave_one_session_out.py --no-gui-select-datasets
```

输出内容包括：

- 每个模型的折级别指标
- 汇总统计
- 混淆矩阵
- 模型参数摘要
- `species/task/session` 等元数据分层分析
- 不同训练集比例（如 `10/30/50/100%`）下的方法对比
- 多 session 是否允许直接合并训练的条件分组结果
- 可通过 GUI 交互式选择数据集路径
- cross-session leave-one-session-out 泛化结果与自动可视化

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

## Cross-Session 实验

对于 cross-session 联合实验，建议只在以下条件同时满足时将 session 视为同组：

- `Monkey` 一致
- `Slot` 一致
- `Task` 一致
- `nTargets` 一致

为此提供了 [group_sessions_by_condition.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/scripts/group_sessions_by_condition.py:1)。
它会：

- 兼容 `Session/Run/Monkey/Slot/Task/nTargets` 这类描述文件字段
- 支持通过 `dataset_root` 自动匹配 `rt_fUS_data_S*_R*.mat`
- 按 `Monkey + Slot + Task + nTargets` 生成 merge-safe 分组
- 标记哪些组具备 cross-session leave-one-session-out 的最小条件
- 单独列出缺失条件或找不到数据文件的 session
- 支持通过 GUI 选择多个 session 文件或 dataset 文件夹后自动分组

[cross_session_leave_one_session_out.py](/Users/ibuprofen/Desktop/fus-bmi/rt_fUS_BMI/py_fus_decoder/scripts/cross_session_leave_one_session_out.py:1) 会在每个组内执行 leave-one-session-out：

- `train = all sessions except held-out session`
- `test = held-out session`
- 循环遍历组内每个 session，并输出每个 fold 的 `accuracy`

为避免数据泄露，cross-session 评测中采用：

- 每个 session 独立 z-score normalization，消除 session 间整体偏移
- 训练 session 与测试 session 的 normalization 统计彼此独立
- `PCA/cPCA/LDA` 仅在训练集上拟合

cross-session 输出包括：

- 每个 fold 的 `accuracy`
- 平均 `accuracy`
- confusion matrix
- classification report
- 不同模型的 cross-session accuracy 柱状图
- confusion matrix 热力图

这让后续继续扩展：

- 新物种数据适配器
- 新任务标签体系
- 新模型结构
- 新指标与可解释性分析

都只需要修改局部模块
