from loguru import logger
import torch
from typing import List
import torch.nn.functional as F
from einops import rearrange

def is_choice(choice: str, match: str) -> bool:
    # Many tokenizers don't just use Yes, but \nYes, " Yes" "ĠYes" "###Yes" and so on. We need to catch all variants. This will also catch eyes, but it's a minor problem, it's unlikely to be a likely token.
    return (match.lower().endswith(choice) or match.lower().startswith(choice)) and len(
        match
    ) < len(choice) + 4


def get_choice_ids(tokenizer, positive_word="yes", negative_word="no") -> List[List[int]]:
    """Get token IDs for Yes/No choices - returns [negative_ids, positive_ids]."""
    positive_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice(positive_word, k)}
    negative_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice(negative_word, k)}
    return [list(negative_choices.values()), list(positive_choices.values())]


def calc_nll(input_ids, logits, attention_mask):
    """Calculate per-sequence NLL from input_ids and logits.
    
    Uses F.cross_entropy_loss with ignore_index=-100 for efficient masking.
    PyTorch skips computation for masked tokens internally.
    """
    shift_logits = logits[:, :-1, :]  # [b, s-1, vocab]
    shift_labels = input_ids[:, 1:]  # [b, s-1]
    shift_mask = attention_mask[:, 1:]  # [b, s-1]
    
    # Mask padded tokens with -100 (ignored by loss function)
    shift_labels = torch.where(shift_mask == 1, shift_labels, -100)
    
    b, s = shift_labels.shape
    
    # F.cross_entropy_loss with reduction='none' gives per-token loss
    # ignore_index=-100 means masked tokens are skipped (return 0 loss)
    token_nll = F.cross_entropy(
        rearrange(shift_logits, 'b s v -> (b s) v'),
        rearrange(shift_labels, 'b s -> (b s)'),
        reduction='none',
        ignore_index=-100
    ).view(b, s)
    
    # Sum over sequence, divide by number of valid tokens
    seq_nll = token_nll.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
    return seq_nll


def get_choice_logprobs(logits_last, choice_ids):
    """
    Extract log probabilities for each choice group.
    
    Args:
        logits_last: [b, vocab] logits for the last token position
        choice_ids: [n_choices, n_ids_per_choice] token IDs for each choice
    
    Returns:
        logp_choices: [b, n_choices] log probabilities
    """
    logp = logits_last.log_softmax(dim=-1)  # [b, vocab]
    b = logp.shape[0]
    logp_choices = torch.zeros(b, len(choice_ids), device=logp.device)
    
    for i, choice_id_group in enumerate(choice_ids):
        choice_id_group = torch.tensor(choice_id_group, device=logp.device)
        # Sum probabilities for all variants of this choice (e.g., "Yes", " Yes", "\nYes")
        logp_choice = logp[:, choice_id_group].logsumexp(-1)  # [b]
        logp_choices[:, i] = logp_choice
    
    return logp_choices

@torch.no_grad()
def gen_with_choices(model, tokenizer, input_ids, attention_mask, choice_ids, continue_n_tokens=0, warn_low_pmass=True):
    """
    Generate one token and extract choice logprobs, optionally continue generating.
    
    Simple approach: format prompt to end at choice point (e.g., "My choice:"),
    run forward pass, extract choice logprobs from next token position.
    
    **Tokenization caveat**: We generate only 1 token. Different models tokenize differently - 
    some might need [" ", "Yes"], ["Ye", "s"], or ["\nYes"]. If your choices have low prob mass 
    (< 10% of max token prob), you'll get NaN in logratios. Edit the message format in 
    apply_chat_template to match your model's tokenization.
    
    Args:
        model: LLM
        tokenizer: tokenizer
        input_ids: [b, s] input tokens ending at the choice point (e.g., "My choice:")
        attention_mask: [b, s]
        choice_ids: [n_choices, n_ids_per_choice] token IDs for each choice
        continue_n_tokens: if >0, continue generating this many more tokens using KV cache
        warn_low_pmass: if True, warn when choice prob mass is low (useful for debugging tokenization)
    
    Returns:
        outputs: generation output with sequences, logits, past_key_values
        seq_nll: [b] NLL for input sequence
        logp_choices: [b, n_choices] log probabilities for each choice
        logratios: [b] log(P(positive)/P(negative))
    """
    model.eval()
    
    # Forward pass on inputs to get NLL and KV cache
    out = model(input_ids, attention_mask=attention_mask, use_cache=True)
    seq_nll = calc_nll(input_ids, out.logits, attention_mask)
    
    # Extract choice logprobs from last position
    logp_choices = get_choice_logprobs(out.logits[:, -1], choice_ids)  # [b, n_choices]
    
    # Calculate log ratio (assuming [negative, positive] order)
    logratios = logp_choices[:, 1] - logp_choices[:, 0]
    
    # Mark as nan if choices are less than 10% of max probable token
    # This covers cases where model isn't confident in any choice (e.g., wrong tokenization)
    maxp = out.logits[:, -1].log_softmax(-1).max(-1)[0].exp()  # [b]
    pmass = logp_choices.exp().sum(-1)  # [b]
    low_pmass_mask = pmass < 0.01 * maxp
    
    # Warn on low prob mass (helpful for debugging tokenization issues)
    if warn_low_pmass and low_pmass_mask.any():
        # Get top-k tokens for debugging
        k = 10
        logp = out.logits[:, -1].log_softmax(-1)
        topk_probs, topk_ids = logp.topk(k, dim=-1)
        
        for i in range(input_ids.shape[0]):
            if low_pmass_mask[i]:
                top_tokens = [(tokenizer.decode([tid.item()]), p.exp().item()) 
                              for tid, p in zip(topk_ids[i], topk_probs[i])]
                top_tokens_str = ", ".join([f"{tok!r} ({prob:.2%})" for tok, prob in top_tokens])
                question_s = tokenizer.batch_decode(input_ids)[i]
                
                logger.warning(
                    f"Low choice prob mass: {pmass[i].item():.2%} < 10% of max ({maxp[i].item():.2%}). "
                    # f"Your choices might not match the model's tokenization. OR the model might be refusing or incoherent"
                    f"Top-{k} tokens: {top_tokens_str}. "
                    # f"Consider adjusting the message format in apply_chat_template to match."
                    # f"Question was `{question_s}`"
                )
                break  # Only warn once per batch
    
    logratios = torch.where(low_pmass_mask, float('nan'), logratios)
    
    # Compute entropy H at choice point: H = -sum(p * log(p))
    logp_full = out.logits[:, -1].log_softmax(-1)  # [b, vocab]
    p_full = logp_full.exp()
    H = -(p_full * logp_full).sum(-1)  # [b] entropy in nats
    
    # Start with just the input
    sequences = input_ids
    logits_list = [out.logits]
    kv_cache = out.past_key_values
    
    # Optionally continue generation
    if continue_n_tokens > 0:
        for _ in range(continue_n_tokens):
            # Get next token from previous logits
            next_token = out.logits[:, -1].log_softmax(-1).argmax(-1, keepdim=True)  # [b, 1]
            
            # Update attention mask
            b = input_ids.shape[0]
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(b, 1, dtype=torch.long, device=input_ids.device)
            ], dim=1)
            
            # Continue from KV cache
            cache_len = kv_cache.get_seq_length()
            out = model(
                next_token,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                cache_position=torch.arange(cache_len, cache_len + 1, dtype=torch.long, device=input_ids.device),
                use_cache=True
            )
            
            sequences = torch.cat([sequences, next_token], dim=1)
            logits_list.append(out.logits)
            kv_cache = out.past_key_values
    
    # Package output similar to generate()
    class Output:
        pass
    outputs = Output()
    outputs.sequences = sequences
    outputs.logits = logits_list
    outputs.past_key_values = kv_cache
    outputs.H = H  # entropy at choice point
    
    return outputs, seq_nll, logp_choices, logratios

