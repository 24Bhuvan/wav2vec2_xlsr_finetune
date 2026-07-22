import logging
from typing import Any, Dict, List, Optional, Union
import torch

logger = logging.getLogger("inference")


def align_tokens(
    emissions: torch.Tensor, 
    token_ids: Union[List[int], torch.Tensor], 
    blank_id: int = 0
) -> Optional[List[Dict[str, Any]]]:
    """Computes the optimal time-alignment path through the log-softmax emission matrix 
    using the Viterbi algorithm for Connectionist Temporal Classification (CTC).
    
    This function replaces the legacy torchaudio.functional.forced_alignment implementation 
    manually with a pure PyTorch execution grid following the exact Graves CTC specification.

    Complexity:
        Time Complexity: O(T * S), where T is num_frames and S = 2 * len(token_ids) + 1.
        Space Complexity: O(T * S) for storing the backpointer tracking trellis matrix.

    Args:
        emissions (Tensor): Log-softmax probability matrix of shape (num_frames, vocab_size).
        token_ids (list or Tensor): Sequence of target token IDs from the transcript text.
        blank_id (int): Vocabulary index allocated to the CTC blank character. Default is 0.

    Returns:
        list of dict: A list containing dictionaries matching the expected structure:
                      [{"token_index": int, "frame_index": int, "score": float}, ...]
                      Returns None if alignment is mathematically impossible or inputs fail validation.
    """
    # ==========================================================================
    # 1. INPUT VALIDATION & SHAPE VERIFICATION
    # ==========================================================================
    if emissions is None or not isinstance(emissions, torch.Tensor):
        logger.error("Invalid emissions input: Expected a torch.Tensor instance.")
        return None

    if emissions.ndim != 2:
        logger.error(f"Invalid emissions shape: Expected 2D tensor (T, V), got {emissions.shape}.")
        return None

    num_frames, vocab_size = emissions.shape

    if num_frames == 0 or vocab_size == 0:
        logger.error("Emissions tensor cannot contain empty dimensions.")
        return None

    # Standardize token target sequence into a raw Python integer list
    if isinstance(token_ids, torch.Tensor):
        targets = token_ids.detach().cpu().flatten().tolist()
    elif isinstance(token_ids, list):
        targets = [int(t) for t in token_ids]
    else:
        logger.error("Invalid target format: token_ids must be a list or torch.Tensor.")
        return None

    if len(targets) == 0:
        logger.error("Target token list is empty. Cannot execute forced alignment alignment grids.")
        return None

    # ==========================================================================
    # 2. TRELLIS ARCHITECTURE STRUCTURING
    # ==========================================================================
    num_tokens = len(targets)
    seq_len = 2 * num_tokens + 1

    # Check hard boundary constraint according to original CTC formulation
    if num_frames < num_tokens:
        logger.error(f"Impossible alignment: num_frames ({num_frames}) < targets len ({num_tokens}).")
        return None

    # Map state indexes back to the target index positions
    state_to_token_idx = []
    extended_seq = []
    for i in range(num_tokens):
        extended_seq.extend([blank_id, targets[i]])
        state_to_token_idx.extend([-1, i])
    extended_seq.append(blank_id)
    state_to_token_idx.append(-1)

    # Move operation variables to the correct device matching emission constraints
    device = emissions.device
    extended_seq_t = torch.tensor(extended_seq, dtype=torch.long, device=device)

    # Initialize dynamic programming Viterbi matrices
    trellis = torch.full((num_frames, seq_len), float("-inf"), dtype=emissions.dtype, device=device)
    backpointers = torch.full((num_frames, seq_len), -1, dtype=torch.long, device=device)

    # Standard CTC Initialization: Force alignment to begin exclusively in the initial blank state.
    trellis[0, 0] = emissions[0, extended_seq[0]]
    backpointers[0, 0] = 0

    # ==========================================================================
    # 3. DYNAMIC PROGRAMMING RECURSION GRID (VITERBI FORWARD STEP)
    # ==========================================================================
    for t in range(1, num_frames):
        frame_emissions = emissions[t, extended_seq_t]
        
        # Calculate valid state limits based on time and remaining frame boundaries
        min_state = max(0, seq_len - 2 * (num_frames - t))
        max_state = min(seq_len - 1, 2 * t + 1)

        for s in range(min_state, max_state + 1):
            val_stay = trellis[t - 1, s]
            best_val = val_stay
            best_bp = s

            if s > 0:
                val_prev = trellis[t - 1, s - 1]
                if val_prev > best_val:
                    best_val = val_prev
                    best_bp = s - 1

            if s > 1 and extended_seq[s] != blank_id and extended_seq[s] != extended_seq[s - 2]:
                val_skip = trellis[t - 1, s - 2]
                if val_skip > best_val:
                    best_val = val_skip
                    best_bp = s - 2

            if best_val != float("-inf"):
                trellis[t, s] = best_val + frame_emissions[s]
                backpointers[t, s] = best_bp

    # ==========================================================================
    # 4. PATH TERMINATION DETERMINATION & BACKTRACKING
    # ==========================================================================
    state_choices = [seq_len - 1]
    if seq_len > 1:
        state_choices.append(seq_len - 2)

    best_final_score = float("-inf")
    current_state = -1

    for s in state_choices:
        if trellis[num_frames - 1, s] > best_final_score:
            best_final_score = trellis[num_frames - 1, s]
            current_state = s

    if current_state == -1 or best_final_score == float("-inf"):
        logger.error("Forced alignment path convergence failed: Trellis space is disconnected.")
        return None

    path_states = []
    for t in range(num_frames - 1, -1, -1):
        path_states.append((t, current_state))
        current_state = int(backpointers[t, current_state])
        if current_state == -1 and t > 0:
            logger.error(f"Broken backtracking link detected at frame entry node index: {t}")
            return None

    path_states.reverse()

    # ==========================================================================
    # 5. FORMAT PACKAGING MATCHING TARGET PUBLIC API SPECIFICATION
    # ==========================================================================
    alignment_path = []
    for t, s in path_states:
        token_idx = state_to_token_idx[s]
        score_val = float(emissions[t, extended_seq[s]])
        
        alignment_path.append({
            "token_index": token_idx,
            "frame_index": t,
            "state_index": s,
            "is_blank": (extended_seq[s] == blank_id),
            "score": score_val
        })

    # ==========================================================================
    # PATH DEBUGGING PRINTS
    # ==========================================================================
    print("\n===== PATH DEBUG =====")

    first_char = None
    for p in alignment_path:
        if not p["is_blank"]:
            first_char = p
            break

    print("First non-blank frame :", first_char["frame_index"] if first_char else "None")
    print("First token index     :", first_char["token_index"] if first_char else "None")
    print("First state index     :", first_char["state_index"] if first_char else "None")

    print()

    for p in alignment_path[:40]:
        print(
            p["frame_index"],
            p["state_index"],
            p["token_index"],
            p["is_blank"]
        )

    return alignment_path


def ctc_decode_emissions(
    emissions: torch.Tensor,
    vocabulary: Dict[int, str],
    blank_id: int = 0
) -> Dict[str, Any]:
    """Performs standard Connectionist Temporal Classification (CTC) decoding.
    
    Takes emission log probabilities or logits, extracts the argmax frame tokens, 
    collapsed identical adjacent duplicates, strips away CTC blank tokens, and maps 
    the resulting sequence back into words preserving spaces and apostrophes.

    Args:
        emissions (Tensor): Frame emission distribution matrix of shape (num_frames, vocab_size).
        vocabulary (dict): Index-to-string mapping matching the acoustic model's tokenizer.
        blank_id (int): Vocabulary index assigned to the CTC blank character. Default is 0.

    Returns:
        dict: A production dictionary housing:
              {
                  "text": str,
                  "token_ids": list of int,
                  "alignment_ready_tokens": list of int
              }
    """
    try:
        if emissions is None or not isinstance(emissions, torch.Tensor):
            raise ValueError("Emissions grid must be an instantiated torch.Tensor profile.")

        if emissions.ndim != 2:
            raise ValueError(f"Expected 2D matrix layout (T, V), resolved shape {emissions.shape}.")

        # Step 1: Greedy argmax projection over all frames
        argmax_tokens = torch.argmax(emissions, dim=-1).detach().cpu().tolist()

        # Step 2: Collapse consecutive duplicates and filter out blank tokens
        collapsed_token_ids: List[int] = []
        prev_token: Optional[int] = None

        for token in argmax_tokens:
            if token != prev_token:
                if token != blank_id:
                    collapsed_token_ids.append(token)
                prev_token = token

        # Step 3: Map tokens to their corresponding characters in the vocabulary
        decoded_chars: List[str] = []
        for t_id in collapsed_token_ids:
            char = vocabulary.get(t_id, "")
            if char:
                decoded_chars.append(char)
            else:
                logger.warning(f"Encountered unknown token ID {t_id} missing from tokenizer vocabulary mapping.")

        # Step 4: Reconstitute the text from the character stream
        decoded_text = "".join(decoded_chars).strip()

        return {
            "text": decoded_text,
            "token_ids": collapsed_token_ids,
            "alignment_ready_tokens": collapsed_token_ids
        }

    except Exception as exc:
        logger.error("Failed to execute CTC greedy decoding stack: %s", exc, exc_info=True)
        return {
            "text": "",
            "token_ids": [],
            "alignment_ready_tokens": []
        }
