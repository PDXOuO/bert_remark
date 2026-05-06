import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ==================== 配置 ====================
MODEL_NAME = "./models/bert-base-chinese"
OUTPUT_DIR = "./results_chnsenticorp"

print(f"✅ 当前使用设备: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA 版本: {torch.version.cuda}\n")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1}


def train():
    # 1. 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 2. 加载数据集
    dataset = load_dataset(
        'csv',
        data_files={
            'train': 'data/train.tsv',
            'test': 'data/dev.tsv'
        },
        delimiter="\t",
    )
    print("✅ 数据集加载完成:", dataset)

    # 3. 分词
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)  # 可提升到256

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        report_to="tensorboard",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model("./checkpoint")
    tokenizer.save_pretrained("./checkpoint")
    print("🎉 训练完成！模型保存在 ./checkpoint 文件夹")


if __name__ == "__main__":
    train()