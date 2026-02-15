import os
import pandas as pd
import tiktoken
from src.models.model import AudioAnalyzerAssistant

# --- 1. SYSTEM INITIALIZATION ---
# This loads your actual model weights for local inference
BASE_PATH = os.path.join('.', 'assistantBinaryModel')
BASE_AUDIO_TEST = os.path.join(BASE_PATH, 'test_audio')
GPT_MODEL_PATH = os.path.join(BASE_PATH, "Final_production_ready_model.pt")

print("🔄 INITIALIZING ASSISTANT...")
tokenizer = tiktoken.get_encoding("gpt2")
assistant = AudioAnalyzerAssistant(GPT_MODEL_PATH, tokenizer=tokenizer)
print("✅ MODEL LOADED (Local Environment Ready)\n")

# --- 2. YOUR DEMO TASKS ---
# Using your exact structure to trigger the model's logic bridges
file_paths = {
    "Rain on awning, canopy.wav": os.path.join(BASE_AUDIO_TEST, "Rain on awning, canopy.wav"),
    "Waves on a quiet New Zealand beach.wav": os.path.join(BASE_AUDIO_TEST, "Waves on a quiet New Zealand beach.wav"),
    "crowd2.wav": os.path.join(BASE_AUDIO_TEST, "crowd2.wav")
}

demo_tasks = [
    {
        "path": file_paths.get("Rain on awning, canopy.wav"), 
        "questions": ["Are things getting wet?", "Is the sound of water constant or intermittent?"]
    },
    {
        "path": file_paths.get("Waves on a quiet New Zealand beach.wav"), 
        "questions": ["Is the environment serene?"]
    },
    {
        "path": file_paths.get("crowd2.wav"), 
        "questions": ["Are multiple genders speaking?", "Is the atmosphere lively?"]
    }
]

# --- 3. ACTUAL LOCAL INFERENCE & LOGGING ---
# Itemizing results row-by-row in the terminal
print("█" * 80)
print("🚀 RUNNING LOCAL INFERENCE AUDIT")
print("█" * 80 + "\n")

all_audit_results = []

for task in demo_tasks:
    file_name = os.path.basename(task["path"])
    
    # LIVE CALL: The assistant actually processes the audio now
    # We pass the single task to the analyzer
    results = assistant.analyze_audio([task]) 
    
    for res in results:
        # Standardized Terminal Output Labels
        print(f"📄 FILE      : {file_name}")
        print(f"❓ QUESTION  : {res.get('Question')}")
        print(f"🤖 RESPONSE  : {res.get('AI Reasoning')}")
        print("-" * 80)
        
        all_audit_results.append({
            "FILE": file_name,
            "QUESTION": res.get('Question'),
            "MODEL RESPONSE": res.get('AI Reasoning')
        })

# --- 4. FINAL ORGANIZED SUMMARY TABLE ---
# Clean Pandas summary for the terminal audit
print("\n📊 FINAL ORGANIZED AUDIT SUMMARY")
print("=" * 80)
summary_df = pd.DataFrame(all_audit_results)

# Setting colwidth to avoid text clipping in terminal
pd.set_option('display.max_colwidth', None)
print(summary_df.to_string(index=False, justify='left'))

print("\n" + "█" * 80)
print("✅ EXECUTION COMPLETE")