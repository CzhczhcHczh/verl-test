# VLM Benchmark Evaluation Suite (14 Benchmarks)

本目录包含 14 个视觉语言模型(VLM)评测基准的完整评测脚本，适用于 Qwen2.5-VL 等 VLM 模型。

## 目录结构

```
verl-test/
├── examples/
│   ├── data_preprocess/          # 数据预处理脚本（HF → Parquet）
│   │   ├── chartqa.py
│   │   ├── cvbench.py
│   │   ├── dynamath.py
│   │   ├── emma.py
│   │   ├── logicvista.py
│   │   ├── vstar_bench.py
│   │   ├── muirbench.py
│   │   ├── vizwiz_vqa.py
│   │   ├── blink.py
│   │   ├── flickr30k.py
│   │   ├── pope.py
│   │   ├── gqa.py
│   │   ├── seedbench.py
│   │   └── textvqa.py
│   └── eval/                     # 评测脚本
│       ├── utils/                # 共享工具模块
│       │   ├── model_utils.py    # 模型加载 & 多模态推理
│       │   └── common.py         # 通用评分/提取函数
│       ├── run_all_benchmarks.sh # 一键运行全部14个benchmark
│       ├── requirements.txt      # Python依赖
│       ├── chartqa/              # ChartQA 评测
│       ├── cvbench/              # CV-Bench 评测
│       ├── vstar_bench/          # V*Bench 评测
│       ├── muirbench/            # MuirBench 评测
│       ├── blink/                # BLINK 评测
│       ├── seedbench/            # SEED-Bench 评测
│       ├── pope/                 # POPE 评测
│       ├── gqa/                  # GQA 评测
│       ├── textvqa/              # TextVQA 评测
│       ├── vizwiz_vqa/           # VizWiz-VQA 评测
│       ├── dynamath/             # DynaMath 评测
│       ├── emma/                 # EMMA 评测
│       ├── logicvista/           # LogicVista 评测
│       └── flickr30k/            # Flickr30k 评测
└── verl/
    └── utils/
        └── reward_score/         # VERL RL reward 计算模块
            ├── chartqa.py
            ├── cvbench.py
            ├── vstar_bench.py
            ├── muirbench.py
            ├── blink.py
            ├── seedbench.py
            ├── pope.py
            ├── gqa.py
            ├── textvqa.py
            ├── vizwiz_vqa.py
            ├── dynamath.py
            ├── emma.py
            ├── logicvista.py
            └── flickr30k.py
```

## 14 个 Benchmark 一览

| # | Benchmark | HF Dataset | 类型 | 核心指标 | 一句话介绍 |
|---|-----------|-----------|------|---------|-----------|
| 1 | **ChartQA** | `lmms-lab/ChartQA` | 短答开放题 | relaxed_accuracy | 图表理解问答，5%数值容差 |
| 2 | **CV-Bench** | `nyu-visionx/CV-Bench` | 多选 | accuracy | 计算机视觉能力评测 |
| 3 | **V\*Bench** | `lmms-lab/vstar-bench` | 多选(ABCD) | accuracy | 视觉搜索与推理 |
| 4 | **MuirBench** | `MUIRBENCH/MUIRBENCH` | 多选(多图) | accuracy | 多图理解与推理 |
| 5 | **BLINK** | `BLINK-Benchmark/BLINK` | 多选(多图) | accuracy | 多图视觉推理 |
| 6 | **SEED-Bench** | `lmms-lab/SEED-Bench` | 多选(ABCD) | accuracy | 综合视觉理解(仅图像题) |
| 7 | **POPE** | `lmms-lab/POPE` | Yes/No | acc/precision/recall/F1 | 物体幻觉检测 |
| 8 | **GQA** | `lmms-lab/GQA` | 短答开放题 | exact_match | 视觉推理问答 |
| 9 | **TextVQA** | `lmms-lab/textvqa` | 短答开放题 | vqa_accuracy | 图像中文字理解 |
| 10 | **VizWiz-VQA** | `lmms-lab/VizWiz-VQA` | 短答开放题 | vqa_accuracy | 盲人拍摄图像问答 |
| 11 | **DynaMath** | `kcz358/DynaMath` | 数学推理(CoT) | avg_acc/worst_acc | 动态数学推理(含变体分析) |
| 12 | **EMMA** | `lmms-lab/EMMA` | 数学/科学 | emma_score | 多模态数学科学推理(含符号比较) |
| 13 | **LogicVista** | `lscpku/LogicVista` | 逻辑推理(CoT) | acc_score | 视觉逻辑推理 |
| 14 | **Flickr30k** | `lmms-lab/flickr30k` | Image Captioning | CIDEr/BLEU/METEOR/ROUGE | 图像描述生成 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r examples/eval/requirements.txt
```

### 2. 运行单个 Benchmark

每个 benchmark 目录下有独立的 `run_xxx_eval.sh`，可通过环境变量指定模型路径：

```bash
# 例：运行 ChartQA
export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export LORA_PATH="/path/to/lora/adapter"  # 可选
export OUTPUT_DIR="results/chartqa"
bash examples/eval/chartqa/run_chartqa_eval.sh
```

### 3. 一键运行全部 14 个 Benchmark

```bash
export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
export LORA_PATH="/path/to/lora/adapter"  # 可选
export RESULT_ROOT="results"
bash examples/eval/run_all_benchmarks.sh
```

运行完成后会在 `RESULT_ROOT` 目录下生成 `summary_<timestamp>.txt` 汇总报告。

### 4. 数据预处理（可选）

如需将数据预处理为 VERL Parquet 格式：

```bash
python examples/data_preprocess/chartqa.py --local_save_dir ~/data/chartqa
python examples/data_preprocess/gqa.py --local_save_dir ~/data/gqa
# ... 其他 benchmark 同理
```

## 各 Benchmark 脚本详情

### ChartQA
- **数据预处理**: `examples/data_preprocess/chartqa.py` — 加载 HF 数据集并转为 Parquet
- **评测脚本**: `examples/eval/chartqa/eval_chartqa.py` — 推理+relaxed accuracy评分(5%数值容差)
- **运行脚本**: `examples/eval/chartqa/run_chartqa_eval.sh`
- **Reward**: `verl/utils/reward_score/chartqa.py`

### CV-Bench
- **数据预处理**: `examples/data_preprocess/cvbench.py`
- **评测脚本**: `examples/eval/cvbench/eval_cvbench.py` — 多选字母匹配
- **运行脚本**: `examples/eval/cvbench/run_cvbench_eval.sh`
- **Reward**: `verl/utils/reward_score/cvbench.py`

### V*Bench
- **数据预处理**: `examples/data_preprocess/vstar_bench.py`
- **评测脚本**: `examples/eval/vstar_bench/eval_vstar_bench.py` — 选项字母匹配
- **运行脚本**: `examples/eval/vstar_bench/run_vstar_bench_eval.sh`
- **Reward**: `verl/utils/reward_score/vstar_bench.py`

### MuirBench
- **数据预处理**: `examples/data_preprocess/muirbench.py` — 支持多图输入
- **评测脚本**: `examples/eval/muirbench/eval_muirbench.py` — 多图推理+大小写不敏感匹配
- **运行脚本**: `examples/eval/muirbench/run_muirbench_eval.sh`
- **Reward**: `verl/utils/reward_score/muirbench.py`

### BLINK
- **数据预处理**: `examples/data_preprocess/blink.py` — 支持多图(image_1~4)
- **评测脚本**: `examples/eval/blink/eval_blink.py` — 多图推理+字母匹配
- **运行脚本**: `examples/eval/blink/run_blink_eval.sh`
- **Reward**: `verl/utils/reward_score/blink.py`

### SEED-Bench
- **数据预处理**: `examples/data_preprocess/seedbench.py` — 仅保留图像类问题(category 1-9)
- **评测脚本**: `examples/eval/seedbench/eval_seedbench.py` — 选项字母匹配
- **运行脚本**: `examples/eval/seedbench/run_seedbench_eval.sh`
- **Reward**: `verl/utils/reward_score/seedbench.py`

### POPE
- **数据预处理**: `examples/data_preprocess/pope.py`
- **评测脚本**: `examples/eval/pope/eval_pope.py` — Yes/No匹配 + precision/recall/F1
- **运行脚本**: `examples/eval/pope/run_pope_eval.sh`
- **Reward**: `verl/utils/reward_score/pope.py`

### GQA
- **数据预处理**: `examples/data_preprocess/gqa.py` — 合并 instructions 和 images 子集
- **评测脚本**: `examples/eval/gqa/eval_gqa.py` — 大小写/标点不敏感精确匹配
- **运行脚本**: `examples/eval/gqa/run_gqa_eval.sh`
- **Reward**: `verl/utils/reward_score/gqa.py`

### TextVQA
- **数据预处理**: `examples/data_preprocess/textvqa.py`
- **评测脚本**: `examples/eval/textvqa/eval_textvqa.py` — EvalAI标准化 + VQA-style多答案投票
- **运行脚本**: `examples/eval/textvqa/run_textvqa_eval.sh`
- **Reward**: `verl/utils/reward_score/textvqa.py`

### VizWiz-VQA
- **数据预处理**: `examples/data_preprocess/vizwiz_vqa.py`
- **评测脚本**: `examples/eval/vizwiz_vqa/eval_vizwiz_vqa.py` — 同TextVQA评分方式
- **运行脚本**: `examples/eval/vizwiz_vqa/run_vizwiz_vqa_eval.sh`
- **Reward**: `verl/utils/reward_score/vizwiz_vqa.py`

### DynaMath
- **数据预处理**: `examples/data_preprocess/dynamath.py`
- **评测脚本**: `examples/eval/dynamath/eval_dynamath.py` — CoT推理 + `<answer>`/`\boxed{}` 提取 + 变体聚合
- **运行脚本**: `examples/eval/dynamath/run_dynamath_eval.sh`
- **Reward**: `verl/utils/reward_score/dynamath.py`

### EMMA
- **数据预处理**: `examples/data_preprocess/emma.py`
- **评测脚本**: `examples/eval/emma/eval_emma.py` — `\boxed{}` 提取 + latex2sympy符号等价 + word2number
- **运行脚本**: `examples/eval/emma/run_emma_eval.sh`
- **Reward**: `verl/utils/reward_score/emma.py`

### LogicVista
- **数据预处理**: `examples/data_preprocess/logicvista.py`
- **评测脚本**: `examples/eval/logicvista/eval_logicvista.py` — CoT推理 + `<answer>` 提取 + 精确匹配
- **运行脚本**: `examples/eval/logicvista/run_logicvista_eval.sh`
- **Reward**: `verl/utils/reward_score/logicvista.py`

### Flickr30k
- **数据预处理**: `examples/data_preprocess/flickr30k.py`
- **评测脚本**: `examples/eval/flickr30k/eval_flickr30k.py` — pycocoevalcap计算 CIDEr/BLEU/METEOR/ROUGE
- **运行脚本**: `examples/eval/flickr30k/run_flickr30k_eval.sh`
- **Reward**: `verl/utils/reward_score/flickr30k.py`

## 共享工具模块

- `examples/eval/utils/model_utils.py`: 模型加载（支持LoRA）、多模态消息构建、推理生成
- `examples/eval/utils/common.py`: 通用函数库，包含：
  - `extract_boxed()` / `extract_answer_tag()` — 答案提取
  - `parse_choice_from_response()` — 多选项解析
  - `relaxed_correctness()` — ChartQA数值容差
  - `vqa_accuracy()` / `vqa_eval_ai_processor()` — VQA-style评分
  - `extract_answer_letter()` — 选项字母提取
  - `extract_yes_no_simple()` — POPE Yes/No提取
  - `reasoning_extract_and_compare()` — CoT推理答案比较

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 基础模型路径 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `LORA_PATH` | LoRA适配器路径（可选） | 空 |
| `OUTPUT_DIR` | 单benchmark输出目录 | `results/<benchmark>` |
| `RESULT_ROOT` | 统一运行时的根输出目录 | `results` |
| `MAX_NEW_TOKENS` | 最大生成token数 | 因benchmark而异 |
| `TEMPERATURE` | 采样温度 | `0.0` (greedy) |
