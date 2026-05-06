# main.py (仅修改 predict 函数部分，其他保持不变)
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import uvicorn
import time  # 导入 time 库

app = FastAPI(title="情感分析系统 API")


MODEL_PATH = "./checkpoint"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


class TextData(BaseModel):
    text: str


@app.post("/predict")
async def predict(data: TextData):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    start_time = time.time()  # 记录开始时间

    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(outputs.logits, dim=1).item()

    end_time = time.time()  # 记录结束时间
    infer_time = round((end_time - start_time) * 1000, 2)  # 计算耗时(毫秒)

    # 返回更丰富的数据
    return {
        "status": "success",
        "data": {
            "label": "正面" if prediction == 1 else "负面",
            "label_id": prediction,
            "confidence": float(probs[prediction]),
            "prob_pos": float(probs[1]),
            "prob_neg": float(probs[0]),
            "infer_time_ms": infer_time,
            "text_len": len(data.text)
        }
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)