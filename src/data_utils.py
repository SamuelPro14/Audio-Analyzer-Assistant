"""
Multimodal Data Ingestion & Dynamic Batch Alignment
---------------------------------------------------
This module implements the specialized data pipeline required to feed the 
Dual-Stream Transformer with synchronized Audio-Text pairs.

Key Mechanisms:
1. **The Fusion Dataset (AudioReasoningDataset)**:
   - Acts as the 'zipper' between the high-level reasoning logic (JSON) and 
     the pre-computed sensory data (CLAP Embeddings).
   - Filters the dataset in O(1) time to ensure 100% data integrity before training starts.

2. **Causal Alignment Strategy (The Collate Function)**:
   - Implements the critical 'N+1' Label Shift required for Teacher Forcing.
   - Since Audio is injected as the 'Prefix' (Token 0), the text labels must be 
     offset to ensure the model predicts the *next* token, not the current one.

3. **Smart Loss Masking**:
   - Dynamically applies 'ignore_index (-100)' to padding tokens in the label set.
   - This ensures the gradients are calculated ONLY on valid reasoning tokens, 
     preventing the model from wasting capacity on learning empty padding.
"""

from torch.utils.data import Dataset
import torch
import json


class AudioReasoningDataset(Dataset):
    def __init__(self, json_path, features_path, tokenizer):
        # Load logic (The Brain) and audio vectors (The Ears)
        with open(json_path, 'r') as f:
            all_logic = json.load(f)

        # features_path contains the 1024-dim CLAP vectors
        self.features = torch.load(features_path)
        self.tokenizer = tokenizer

        # Filter samples to ensure we only use audios we successfully excavated
        self.active_data = [
            item for item in all_logic
            if item['file_name'] in self.features
        ]
        print(f"🚀 Dataset verified: {len(self.active_data)} samples ready.")

    def __len__(self):
        return len(self.active_data)

    def __getitem__(self, idx):
        sample = self.active_data[idx]

        # Pull 1024-dim audio feature
        audio_tensor = self.features[sample['file_name']].squeeze(0)

        # Apply the "Instruction Fine-Tuning" template
        # Explicitly append <|endoftext|> so the model learns a termination signal
        full_text = (
            f"Below is an instruction describing an audio task. "
            f"Respond appropriately using the provided audio input.\n\n"
            f"### Instruction:\n{sample['question']}\n\n"
            f"### Response:\n{sample['thought_answer']}<|endoftext|>"
        )

        # Convert to IDs without padding yet (Collate will handle batch padding)
        token_ids = self.tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})

        return {
            "audio_input": audio_tensor,
            "token_ids": token_ids
        }

# 2. THE CUSTOM COLLATE FUNCTION (The Tailor)
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100):
    # Stack 1024-dim audio embeddings [Batch, 1024]
    audio_embeds = torch.stack([torch.tensor(item["audio_input"]) for item in batch])

    token_lists = [item["token_ids"] for item in batch]
    batch_max_length = max(len(t) for t in token_lists)

    input_ids_lst, labels_lst = [], []

    for tokens in token_lists:
        # --- THE MULTIMODAL SHIFT ---
        # input_ids: [T0, T1, ..., Tn-1] (Length N)
        # We pad to the batch maximum text length
        input_padded = tokens + [pad_token_id] * (batch_max_length - len(tokens))

        # labels: [T0, T1, ..., Tn] (Length N+1)
        # We add +1 to the length so that the prepended Audio (at index 0)
        # has a target (T0), and the final text token has a target (EOS).
        labels_padded = tokens + [pad_token_id] * (batch_max_length + 1 - len(tokens))

        input_ids = torch.tensor(input_padded)
        labels = torch.tensor(labels_padded)

        # --- SMART MASKING ---
        # We keep the VERY FIRST pad_token_id as the 'Tend' (Stop Signal) target.
        # Since labels is length N+1, the stop signal is at index len(tokens).
        mask_start = len(tokens) + 1
        if mask_start < len(labels):
            labels[mask_start:] = ignore_index

        input_ids_lst.append(input_ids)
        labels_lst.append(labels)

    return {
        "audio_embeds": audio_embeds, # Moved to A100 in training loop
        "input_ids": torch.stack(input_ids_lst), # Shape: [Batch, N]
        "labels": torch.stack(labels_lst)        # Shape: [Batch, N+1]
    }