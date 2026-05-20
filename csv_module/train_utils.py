"""
Training utilities for CSV.
Adapted from TSV/train_utils.py.
"""

import torch
import torch.nn.functional as F


def collate_fn(prompts, labels, pad_id: int = 0):
    """
    Pad a list of tokenized prompts to the same length (right-padding,
    matching TSV's original implementation).

    Args:
        prompts: list of tensors, each shape [1, seq_len_i]
        labels:  list of int labels
        pad_id:  token id to use for padding (default 0). Should match
                 tokenizer.pad_token_id for the model in use.

    Returns:
        prompts_padded: [batch_size, max_seq_len]
        labels_tensor:  [batch_size] long tensor
        attention_mask: [batch_size, max_seq_len] long (1=real, 0=pad)
    """
    max_seq_len = max(prompt.size(1) for prompt in prompts)
    batch_size = len(prompts)
    dtype = prompts[0].dtype

    prompts_padded = torch.full((batch_size, max_seq_len), pad_id, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    for i, prompt in enumerate(prompts):
        seq_len = prompt.size(1)
        prompts_padded[i, :seq_len] = prompt.squeeze(0)
        attention_mask[i, :seq_len] = 1

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return prompts_padded, labels_tensor, attention_mask


def get_last_non_padded_token_rep(hidden_states, attention_mask):
    """
    Get the last non-padded token's representation for each sequence
    using vectorized indexing. Assumes RIGHT padding.

    Args:
        hidden_states:  [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len] (or [batch_size, 1, seq_len])

    Returns:
        [batch_size, hidden_size]
    """
    mask = attention_mask
    if mask.dim() == 3:
        mask = mask.squeeze(1)

    lengths = mask.sum(dim=1).long()  # [B]
    batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_idx, lengths - 1]  # [B, D]


def compute_ot_loss_cos(last_token_rep, centroids, pseudo_label, cos_temp):
    """
    Compute vMF-style classification loss.

    Args:
        last_token_rep: [batch_size, hidden_size]
        centroids:      [num_classes, hidden_size]
        pseudo_label:   [batch_size, num_classes] one-hot
        cos_temp:       float (e.g. 0.1)

    Returns:
        loss, similarities
    """
    last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
    centroids_norm = F.normalize(centroids.float(), p=2, dim=-1)

    similarities = torch.matmul(last_token_rep, centroids_norm.T)
    similarities = similarities / cos_temp

    pt = F.softmax(similarities, dim=-1)
    loss = -torch.sum(pseudo_label.float() * torch.log(pt + 1e-8)) / pseudo_label.shape[0]

    return loss, similarities


def update_centroids_ema_hard(centroids, last_token_rep, pseudo_label, ema_decay):
    """
    EMA update of centroids using hard (one-hot) labels.
    All computation in fp32 for numerical stability.

    Args:
        centroids:      [num_classes, hidden_size]
        last_token_rep: [batch_size, hidden_size]
        pseudo_label:   [batch_size, num_classes] one-hot
        ema_decay:      float (e.g. 0.99)

    Returns:
        updated centroids [num_classes, hidden_size] (fp32)
    """
    last_token_rep_norm = F.normalize(last_token_rep.float(), p=2, dim=1)
    centroids_f = F.normalize(centroids.float(), p=2, dim=1)

    max_indices = torch.argmax(pseudo_label, dim=1)
    discrete_labels = torch.zeros_like(pseudo_label, dtype=torch.float32)
    discrete_labels[torch.arange(pseudo_label.size(0)), max_indices] = 1

    weighted_sum = torch.matmul(discrete_labels.T, last_token_rep_norm)
    pseudo_label_sum = discrete_labels.sum(dim=0).unsqueeze(1) + 1e-8
    new_centroids_batch = weighted_sum / pseudo_label_sum

    updated_centroids = F.normalize(
        ema_decay * centroids_f + (1 - ema_decay) * new_centroids_batch,
        p=2, dim=-1
    )
    return updated_centroids