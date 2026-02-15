"""
Inference Logic & Token Sampling Strategies
-------------------------------------------
This module implements the high-performance decoding loop required for 
real-time Auditory Reasoning.

Key Optimizations:
1. **Multimodal KV-Caching**: 
   - Uniquely handles 'Audio Injection' only at Step 0, then seamlessly switches 
     to standard text caching for subsequent tokens. This reduces inference 
     latency from O(N^2) to O(N).

2. **Nucleus & Top-K Filtering**: 
   - Implements robust probability truncation to prevent "tail-end" hallucinations 
     while maintaining the creative variance needed for natural language generation.

3. **Repetition Penalty (The 'Loop Killer')**: 
   - Dynamically penalizes tokens that have already been generated, preventing 
     the model from getting stuck in infinite "audio audio audio" loops—a common 
     issue in small multimodal models.
"""

import torch
import torch.nn.functional as F


def text_to_token_ids(text, tokenizer):
    """
    Utility: Encodes text while explicitly enabling the <|endoftext|> stop signal.
    Essential for 'tiktoken' based tokenizers used in GPT-2.
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def top_k_top_p_filtering(logits, top_k=None, top_p=0.9):
    """
    Applies 'Nucleus Sampling' (Top-P) and 'Top-K' filtering to the raw model logits.

    Mechanism:
    - Sorts probabilities to find the smallest set of tokens that sum to 'top_p' (e.g., 0.9).
    - Masks out all other tokens by setting their probability to -infinity.
    - Preserves tensor dimensionality to prevent runtime shape errors on the GPU.
    """
    B, V = logits.shape

    # 1. Top-K Filtering
    if top_k is not None and 0 < top_k < V:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits.masked_fill_(indices_to_remove, float('-inf'))

    # 2. Top-P (Nucleus) Filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Calculate cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create the mask on the SORTED tensor (stays 2D)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift mask to ensure we keep at least the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Apply the mask to the sorted logits
        sorted_logits.masked_fill_(sorted_indices_to_remove, float('-inf'))

        # RE-SCATTER: Put the masked values back into their original vocabulary positions
        # This ensures the dimensionality [B, V] is preserved perfectly
        logits = torch.full_like(logits, float('-inf')).scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


@torch.no_grad()
def generate_multimodal(
  model, idx, audio_features, max_new_tokens, context_size,
  use_cache=True, temperature=0.8, top_k=None, top_p=0.9,
  repetition_penalty=1.5, # VITAL: Prevents the <||||> loop
  eos_id=50256
):
    """
    The Core Inference Loop: Bridges Audio Perception with Text Generation.

    Key Logic:
    1. **Step 0 Injection**: The 'audio_features' are passed ONLY on the first step.
    2. **KV-Cache Maintenance**: Subsequent steps pass 'None' for audio, relying on 
        the model's internal cache to 'remember' the sound.
    3. **Penalty Application**: Actively scans 'generated_tokens' to penalize repetition,
        forcing the model to progress its reasoning chain.
    """
    model.eval()
    if use_cache:
        model.reset_cache()

    generated_tokens = [] # Tracks history to apply penalty

    for step in range(max_new_tokens):
        # 1. KV-Cache Optimization:
        # Inject audio ONLY on Step 0. Afterwards, it's stored in memory.
        if use_cache:
            if step == 0:
                idx_cond = idx[:, -context_size:]
                audio_cond = audio_features
            else:
                idx_cond = idx[:, -1:] # Only process the very last word
                audio_cond = None      # Audio is already cached
        else:
            idx_cond = idx[:, -context_size:]
            audio_cond = audio_features

        # 2. Forward Pass
        logits = model(idx_cond, audio_features=audio_cond, use_cache=use_cache)[:, -1, :]

        # 3. Apply Repetition Penalty (The "Loop Killer")
        for token in set(generated_tokens):
            if logits[0, token] > 0:
                logits[0, token] /= repetition_penalty
            else:
                logits[0, token] *= repetition_penalty

        # 4. Sampling
        if temperature <= 0.0:
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        # 5. Append and Check Stop Signal
        generated_tokens.append(next_token_id.item())
        idx = torch.cat((idx, next_token_id), dim=-1)

        if next_token_id.item() == eos_id:
            break

    return idx


def generate_multimodal_sample(model, tokenizer, device, question, audio_key, features_dict):
    """
    A 'Sanity Check' utility that performs a single end-to-end inference pass.
    
    Role in System:
    - Acts as a real-time probe during training to visualize the model's progress.
    - Temporarily switches the model to 'eval' mode to disable Dropout, ensuring 
      deterministic and stable output for inspection.
    - formatting: Enforces the exact JSON-instruction prompt structure used in 
      fine-tuning to prevent distribution shift during validation.

    Key Behaviors:
    - Temperature (0.2): Intentionally low to favor high-confidence, logical reasoning.
    - Repetition Penalty (2.0): Aggressively suppresses loops to force the model 
      to move forward in its 'Chain of Thought'.
    - Auto-Restore: Automatically switches the model back to 'train()' mode 
      after inference so gradients continue to flow correctly.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]

    # Format matches your JSON training data exactly
    prompt = (
        f"Below is an instruction describing an audio task. "
        f"Respond appropriately using the provided audio input.\n\n"
        f"### Instruction:\n{question}\n\n"
        f"### Response:\n"
    )

    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    audio_vector = features_dict[audio_key].to(device)

    # Ensure we use the correct ID for Tiktoken
    tiktoken_eos_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

    with torch.no_grad():
        token_ids = generate_multimodal(
            model=model,
            idx=encoded,
            audio_features=audio_vector,
            max_new_tokens=200,
            context_size=context_size,
            temperature=0.2,         # Adds variety
            top_p=0.9,               # Logical filtering
            repetition_penalty=2.0,  # Force the model to speak English
            eos_id=tiktoken_eos_id
        )

    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(f"\n--- INFERENCE RESULT ({audio_key}) ---\n")
    print(decoded_text)
    model.train()