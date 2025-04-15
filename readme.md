## 🧠 Mental Health Chatbot – Fine-Tuned LLaMA-3 with Gradio UI

This project demonstrates how to fine-tune `meta-llama/Llama-3-1B` on a mental health counseling dataset using LoRA, and deploy it as an interactive chatbot with Gradio.

---

### 📍 Features

- Fine-tuned using the [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) dataset  
- Lightweight training via LoRA  
- Conversational prompt formatting  
- Safe text generation with attention mask and pad token handling  
- Clean Gradio chat UI

---

## 🚀 How to Run in Google Colab

### 🔧 1. Set Up Environment
Add your HF_TOKEN in Google Colab settings

```python
!pip install -q transformers peft accelerate gradio bitsandbytes
```

> ⚠️ `bitsandbytes` is required for efficient 8-bit model loading.

---

### 📁 2. Load Tokenizer and Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model & LoRA fine-tuned adapter
base_model_path = "meta-llama/Llama-3.2-1B"
peft_model_path = "./mental_health_finetuned"  # Replace with your path if saved elsewhere

tokenizer = AutoTokenizer.from_pretrained(peft_model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_path)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
```

---

### 💬 3. Define Chat Function

```python
def safe_generate(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
```

---

### 🌐 4. Launch Gradio Interface

```python
import gradio as gr

def chat(user_input, history):
    if history is None:
        history = []

    formatted_prompt = "<|begin_of_text|>"
    for user_msg, bot_msg in history:
        formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>\n"
        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{bot_msg}<|eot_id|>\n"
    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

    response = safe_generate(model, tokenizer, formatted_prompt)
    history.append((user_input, response.strip()))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Mental Health AI Chatbot (LLaMA-3 Fine-Tuned)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask your mental health question here...", label="Your message")
    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    gr.Button("Clear").click(lambda: ([], []), None, [chatbot, state])

demo.launch()
```

---

### ✅ Example Prompt

```text
I feel anxious all the time. What should I do?
```

---

## 📌 Notes

- Your fine-tuned model folder (`mental_health_finetuned`) should include:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - `tokenizer.json`, `tokenizer_config.json`, etc.

- To upload this folder to Colab:
  - Use the Colab file browser to upload
  - Or mount Google Drive

---