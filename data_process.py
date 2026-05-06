import pandas as pd
from datasets import Dataset, DatasetDict
import os

# ==================== 配置 ====================
DATA_DIR = "data"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在读取原始数据...")

# 1. 读取原始文件
train_df = pd.read_csv(f"{DATA_DIR}/train.tsv", delimiter="\t")
dev_df = pd.read_csv(f"{DATA_DIR}/dev.tsv", delimiter="\t")
test_df = pd.read_csv(f"{DATA_DIR}/test.tsv", delimiter="\t")

print(f"原始 train shape: {train_df.shape}")
print(f"原始 dev shape: {dev_df.shape}")
print(f"原始 test shape: {test_df.shape}")

# 2. 查看列名
print("\ntrain columns:", train_df.columns.tolist())
print("dev columns:", dev_df.columns.tolist())
print("test columns:", test_df.columns.tolist())


# 3. 统一列名并清洗
def clean_df(df, has_label=True):
    # 删除 qid 列（如果存在）
    if 'qid' in df.columns:
        df = df.drop(columns=['qid'])

    # 重命名列（适配不同命名）
    if 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})
    if 'polarity' in df.columns:
        df = df.rename(columns={'polarity': 'label'})

    # 确保只有 text 和 label 两列
    if has_label:
        df = df[['label', 'text']]
    else:
        df = df[['text']].copy()
        df['label'] = -1  # test集没有label，统一用 -1 表示

    # 去重和空值处理
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


# 4. 清洗三个数据集
train_clean = clean_df(train_df, has_label=True)
dev_clean = clean_df(dev_df, has_label=True)
test_clean = clean_df(test_df, has_label=False)

print(f"\n清洗后 train: {len(train_clean)} 条")
print(f"清洗后 dev: {len(dev_clean)} 条")
print(f"清洗后 test: {len(test_clean)} 条")

# 5. 保存为 tsv
train_clean.to_csv(f"{OUTPUT_DIR}/train.tsv", sep="\t", index=False)
dev_clean.to_csv(f"{OUTPUT_DIR}/dev.tsv", sep="\t", index=False)  # 作为验证集
test_clean.to_csv(f"{OUTPUT_DIR}/test.tsv", sep="\t", index=False)

print(f"\n✅ 数据清洗完成！文件已保存至 {OUTPUT_DIR}/ 目录")
