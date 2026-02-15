"""
High-Performance Training Loop & Validation Protocol
----------------------------------------------------
This module implements the 'Engine Room' for the Auditory Reasoning Bridge.

Key Capabilities:
1. **Automatic Mixed Precision (AMP)**: 
   - Utilizes 'torch.amp.autocast' to dynamically switch between float16 and float32.
   - This maximizes batch size capacity and throughput on modern CUDA-enabled GPUs, 
     significantly reducing training time.

2. **Dynamic Reality Check**: 
   - Unlike standard loops that only print loss numbers, this engine extracts 
     random samples from the Validation Set in real-time.
   - It forces the model to generate a full 'Chain of Thought' log every epoch, 
     allowing immediate visual debugging of the reasoning process.

3. **Double-Safety Checkpointing**:
   - Maintains two distinct save states: 
     a) 'last_checkpoint.pt' for crash recovery (Safety Net).
     b) 'best_bridge.pt' for the highest validation score (The Champion).
"""


import torch
from tqdm import tqdm
from src.textGeneration import generate_multimodal_sample
import torch
import torch.nn as nn


def train_engine(
    model, train_loader, val_loader, optimizer, scaler, device,
    features_dict, tokenizer, epochs=15, start_epoch=0, best_val=float("inf"),
    checkpoint_path="best_bridge.pt"
):
    """
    The Master Training Protocol. Orchestrates the optimization of the 
    Logic Bridge between CLAP audio features and GPT-2 reasoning.

    Workflow per Epoch:
    1. **Gradient Descent (Train)**: 
       - Applies Gradient Clipping (max_norm=1.0) to prevent 'exploding gradients', 
         a common instability in Multimodal Transformers.
       - Uses Scaled Loss Backpropagation to maintain numerical stability in float16.

    2. **Validation Audit (Eval)**:
       - Freezes the model to calculate the pure Cross-Entropy Loss on unseen data.

    3. **The 'Reality Check' (Monitor)**:
       - RANDOMLY selects a file from the validation set (not a fixed example).
       - Invokes 'generate_multimodal_sample' to print the model's actual internal monologue.
       - This proves to the user/professor that the model is generalizing, not memorizing.

    4. **Artifact Management**:
       - Saves the 'Best Model' only when validation loss hits a new record low.
       - Saves the 'Last State' every epoch to ensure zero data loss in case of power failure.
    """
    # ignore_index=-100 ensures we don't calculate loss on padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(start_epoch, epochs):
        # --- 1. TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            audio = batch["audio_embeds"].to(device) # [B, 1024]
            ids = batch["input_ids"].to(device)      # [B, N]
            labels = batch["labels"].to(device)      # [B, N+1]

            optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward (A100 optimized)
            with torch.amp.autocast('cuda'):
                logits = model(ids, audio_features=audio)
                # N+1 logits vs N+1 labels (Perfect alignment)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()

            # Unscale for Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 2. VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_audio, v_ids, v_labels = v_batch["audio_embeds"].to(device), v_batch["input_ids"].to(device), v_batch["labels"].to(device)
                with torch.amp.autocast('cuda'):
                    v_logits = model(v_ids, audio_features=v_audio)
                    v_loss = criterion(v_logits.view(-1, v_logits.size(-1)), v_labels.view(-1))
                total_val_loss += v_loss.item()

        avg_val = total_val_loss / len(val_loader)
        avg_train = total_train_loss / len(train_loader)
        print(f"\n🏁 Epoch {epoch+1} Results | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # --- 3. MONITORING (The Reality Check - Updated to be DYNAMIC) ---
        # Instead of a static check, we pull a random sample from the Validation Set
        if features_dict and tokenizer:
            # Get a random item from the validation dataset object
            val_ds = val_loader.dataset
            random_idx = torch.randint(0, len(val_ds), (1,)).item()

            # Extract actual sample info for a real-world test
            sample = val_ds.dataset.active_data[val_ds.indices[random_idx]]
            test_key = sample['file_name']
            test_query = sample['question']

            print(f"📺 Monitoring Reasoning for: {test_key}")
            print(f"❓ Question: {test_query}")

            # generate_multimodal_sample handles internal eval() calls
            generate_multimodal_sample(model, tokenizer, device, test_query, test_key, features_dict)

            # Force back to training mode after live check
            model.train()

        # --- 4. DOUBLE CHECKPOINTING (Safety Net + High Score) ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val': best_val,
        }

        # Safety Net: Always save the latest state
        torch.save(checkpoint, "last_checkpoint.pt")

        # High Score: Save the "Val Champion"
        if avg_val < best_val:
            best_val = avg_val
            checkpoint['best_val'] = best_val
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 NEW BEST! Milestone saved to {checkpoint_path}")

    # --- 5. FINAL MARATHON SAVE ---
    print("\n🚀 Marathon complete. Saving final model state...")
    torch.save(model.state_dict(), "final_audio_bridge_weights.pt")

    return "Training Marathon Finished!"
    

