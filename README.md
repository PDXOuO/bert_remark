# ChnSentiCorp BERT 情感分析模型

基于 **BERT-base-chinese** 在中文酒店评论数据集 **ChnSentiCorp** 上进行微调的情感二分类项目（正面 / 负面）。

## 项目简介

本项目使用 Hugging Face Transformers 框架，对 `bert-base-chinese` 模型进行微调，实现中文文本的情感极性分类。数据集采用经典的 **ChnSentiCorp**（酒店评论数据集）。

### 主要功能
- 支持 TSV 格式数据训练
- 使用 Trainer API 进行高效训练
- 动态 Padding + 混合精度 (FP16)
- 自动保存最优模型
- 提供准确率和 F1 分数评估

## 环境要求

- Python 3.9+
- CUDA 12.4 / 12.6（推荐使用 GPU 版本）
- NVIDIA 显卡（推荐 8GB+ 显存）

### 依赖安装

```bash
# 使用 uv（推荐）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

uv pip install transformers datasets scikit-learn accelerate tensorboard
或使用 requirements.txt：
Bashpip install -r requirements.txt
数据集准备

下载 ChnSentiCorp 数据集（TSV 格式）
放到 data/ 目录下，命名为：
data/train.tsv
data/test.tsv


文件格式示例（Tab 分隔）：
tsvlabel	text
1	这家酒店位置很好，服务也很棒！
0	房间很小，设施陈旧，不推荐。
快速开始
1. 训练模型
Bashpython train.py
2. 使用 TensorBoard 查看训练曲线
Bashtensorboard --logdir ./logs
模型推理（示例）
你可以新建 inference.py：
Pythonfrom transformers import pipeline

# 加载训练好的模型
classifier = pipeline(
    "sentiment-analysis",
    model="./checkpoint",
    tokenizer="./checkpoint"
)

result = classifier("这家酒店太棒了，下次还来！")
print(result)
训练参数

基础模型：bert-base-chinese
学习率：2e-5
Batch Size：16（训练） / 32（评估）
Epochs：3
最大长度：128
优化器：AdamW + weight decay

预期效果
在 ChnSentiCorp 测试集上通常可以达到：

Accuracy：92% ~ 94%
F1 Score：92% ~ 94%

（具体效果与超参数、随机种子有关）
TODO

 添加更多数据增强
 支持多分类任务
 部署为 Web API（FastAPI + Gradio）
 尝试更大模型（RoBERTa、LLaMA-3 等）

参考资料

Hugging Face Transformers 官方文档
ChnSentiCorp 数据集

## 项目预览
![页面截图](./assets/demo.png)