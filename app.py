import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# Page Setup
# ==============================
st.set_page_config(
    page_title="Phi-3 AI Chatbot",
    layout="wide"
)

st.markdown("""
<style>
.chat-user {
    background: #1f77b4;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.chat-ai {
    background: #262730;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Phi-3 AI Chatbot")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# ==============================
# Load Model
# ==============================
@st.cache_resource()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return tokenizer, model


tokenizer, model = load_model()

# ==============================
# Chat sessions
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================
# Input box
# ==============================
prompt = st.text_input("Type your message:")

# ==============================
# Send button
# ==============================
if st.button("Send") and prompt:
    st.session_state.messages.append(("user", prompt))

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    st.session_state.messages.append(("ai", reply))

# ==============================
# Chat display
# ==============================
chat_box = st.container()

with chat_box:
    for role, msg in st.session_state.messages:

        if role == "user":
            st.markdown(f"<div class='chat-user'>👤 {msg}</div>", unsafe_allow_html=True)

        else:
            st.markdown(f"<div class='chat-ai'>🤖 {msg}</div>", unsafe_allow_html=True)
