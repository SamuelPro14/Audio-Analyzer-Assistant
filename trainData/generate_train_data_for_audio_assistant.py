"""
Data Generation & Multimodal Feature Extraction Pipeline
--------------------------------------------------------
This script serves as the 'Pre-processing Factory' for the Auditory Reasoning System.
It performs two critical upstream tasks required before training the custom GPT-2 model.

Key Operations:
1. **Synthetic Logic Generation (The Teacher)**:
   - Utilizes a massive 70B parameter model (Llama-3) to generate high-quality 
     "Chain of Thought" reasoning traces for 1,500 audio samples.
   - This creates a "Gold Standard" dataset where every answer is backed by 
     explicit analysis of Texture, Pitch, and Dynamics.

2. **Acoustic Vectorization (The Ears)**:
   - Indexes the raw audio library using an 'Entropy-Resistant' matching algorithm 
     to handle messy filenames.
   - Converts raw .wav waveforms into dense 1024-dimensional CLAP embeddings.
   - Saves these vectors to 'audio_features.pt', acting as the immutable sensory 
     input for the training loop.

Why Run This?
- **Distillation**: We cannot run a 70B model live on a small edge device. We use this 
  script to 'distill' its intelligence into a dataset that our smaller GPT-2 can learn from.
- **Efficiency**: Pre-computing CLAP embeddings allows the training loop to run 
  100x faster, as it doesn't need to process raw audio every epoch.
"""

from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from msclap import CLAP
import os
import re
from msclap import CLAP
import random
import json
from tqdm import tqdm
import os
import pandas as pd
import requests  


# 1. Load Llama 3.3 70B (The Teacher)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# --- CORRECTED DOWNLOADER FOR LOCAL MACHINES ---
file_url = "https://zenodo.org/records/6473207/files/clotho_aqa_train.csv?download=1"
file_name = "full_train.csv"

if not os.path.exists(file_name):
    print("📡 File missing! Downloading now via Python...")
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
else:
    print("✅ File found. Ready to load.")
# -----------------------------------------------

full_df = pd.read_csv("full_train.csv")
raw_samples = full_df.to_dict('records')

# FINAL REFINEMENT: Depth with Structural Safety
prompt_style = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert Auditory Scene Analyst. Provide a dense, technical "Chain of Thought" justification.
CRITICAL CONSTRAINTS:
1. Start with <|start_thought|>
2. Analyze **Texture**, **Pitch**, and **Dynamics** in 1-2 sentences each.
3. **Logic Bridge**: Explain the answer based on these features.
4. **Brevity**: Keep the entire thought block under 200 tokens to ensure you reach the end.
5. End with <|end_thought|>
6. Provide the final Answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Audio: {file_name}
Question: {question}
Known Answer: {answer}
Confidence: {confidence}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<|start_thought|>"""

final_data = []

output_file = "train_reasoning.json"
target_total = 1500

# SHUFFLE FOR DIVERSITY 
random.seed(42) 
random.shuffle(raw_samples)

print(f"🚀 Starting the 1000-sample production run...")
print(f"📂 Saving directly to: {output_file}")

for i in tqdm(range(target_total)):
    item = raw_samples[i]
    
    q = item.get('QuestionText', 'What is happening in this audio?')
    a = item.get('answer', 'unknown')
    c = item.get('confidence', 'Maybe') 
    
    # We pre-fill with one <|start_thought|> to guide the model
    prompt = prompt_style.format(
        file_name = item.get('file_name', 'audio.wav'),
        question = q,
        answer = a,
        confidence = c
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # BUG FIX: Increased tokens to 160 to prevent cut-offs
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2) 
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # BUG FIX: Logic to prevent double-tagging
    generated_chunk = response.split("assistant")[-1].strip()
    
    # If the model repeated the tag we pre-filled, we clean it up
    clean_logic = generated_chunk.replace("<|start_thought|><|start_thought|>", "<|start_thought|>")
    if not clean_logic.startswith("<|start_thought|>"):
        clean_logic = "<|start_thought|>" + clean_logic
    
    final_data.append({
        "file_name": item.get('file_name'),
        "question": q,
        "thought_answer": clean_logic
    })
    
    # Save to disk every 50 samples for safety
    if i % 50 == 0:
        with open(output_file, "w") as f:
            json.dump(final_data, f, indent=4)

# Final complete save
with open(output_file, "w") as f:
    json.dump(final_data, f, indent=4)

print(f"✅ Production Complete! Reasoning chains saved to {output_file}")


# 1. SETUP
json_path = "train_reasoning.json"
audio_root = "audio_data"
output_pt = "audio_features.pt"

# 2. BUILDING THE ENTROPY-RESISTANT INDEX
print("📂 Excavating all audio folders...")
global_index = {} # 'normalized_name' -> 'full_path'

def get_pure_alpha(s):
    # Removes extensions, numbers, and symbols to find the core 'meaning'
    name_only = os.path.splitext(s)[0].lower()
    return re.sub(r'[^a-z]', '', name_only)

for root, _, files in os.walk(audio_root):
    for f in files:
        if f.endswith(('.wav', '.WAV')):
            path = os.path.join(root, f)
            # Store multiple keys for the same path to catch all variations
            global_index[f] = path # Exact
            global_index[get_pure_alpha(f)] = path # Pure Alpha

# 3. EXTRACTION
clap_model = CLAP(version='2023', use_cuda=True)
with open(json_path, "r") as f:
    logic_data = json.load(f)

unique_json_files = list(set([item['file_name'] for item in logic_data]))
feature_map = {}

print(f"🚀 Attempting final excavation for {len(unique_json_files)} samples...")
for fname in tqdm(unique_json_files):
    target_alpha = get_pure_alpha(fname)
    
    # Strategy: Exact -> Pure Alpha -> Semantic Substring
    path = global_index.get(fname) or global_index.get(target_alpha)
    
    if not path:
        # Final Stand: Check if the target alpha is a subset of ANY disk name
        for alpha_key, disk_path in global_index.items():
            if target_alpha and len(target_alpha) > 5: # Only match if the name is substantial
                if target_alpha in alpha_key or alpha_key in target_alpha:
                    path = disk_path
                    break

    if path:
        with torch.no_grad():
            emb = clap_model.get_audio_embeddings([path])
            feature_map[fname] = emb.cpu()

# 4. SAVE
torch.save(feature_map, output_pt)
print(f"🎉 FINAL SPRINT COUNT: Saved {len(feature_map)} audio vectors.")