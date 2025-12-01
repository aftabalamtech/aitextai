from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "microsoft/Phi-3-mini-4k-instruct"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Phi-3 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Model loaded ✅")

class Chat(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: Chat):

    inputs = tokenizer(req.message, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    reply = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"reply": reply}
