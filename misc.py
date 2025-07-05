# unused stuff that I don't want to throw away quite yet
def selective_freeze_embedding_layer(model, tokenizer, unfrozen_words):
    """
    Freezes the embedding layer of a transformer model,
    but allows selected tokens (from unfrozen_words) to remain trainable.

    Args:
        model: Hugging Face transformer model (e.g., AutoModel)
        tokenizer: Corresponding tokenizer (e.g., AutoTokenizer)
        unfrozen_words: List or set of words to keep trainable
    """
    # Freeze the entire embedding layer
    embedding_layer = model.embeddings.word_embeddings
    embedding_layer.weight.requires_grad = True  # must stay True for masking
    for param in model.embeddings.parameters():
        param.requires_grad = True  # required for backward hook to work

    # Get token IDs of unfrozen words and all special tokens
    token_ids = set()
    for word in unfrozen_words:
        ids = tokenizer(word, add_special_tokens=False)['input_ids']
        token_ids.update(ids)

    # Add all special token IDs
    if hasattr(tokenizer, "all_special_ids"):
        token_ids.update(tokenizer.all_special_ids)
    else:
        # Fallback for tokenizers without all_special_ids
        for tok in tokenizer.all_special_tokens:
            ids = tokenizer(tok, add_special_tokens=False)['input_ids']
            token_ids.update(ids)

    vocab_size, hidden_size = embedding_layer.weight.shape
    grad_mask = torch.zeros(vocab_size, 1, device=embedding_layer.weight.device)
    for idx in token_ids:
        if idx < vocab_size:
            grad_mask[idx] = 1.0

    # Register gradient hook to zero out updates for frozen tokens
    def hook_fn(grad):
        # grad: [vocab_size, hidden_size]
        return grad * grad_mask

    embedding_layer.weight.register_hook(hook_fn)
    
class WarmupInverseSquareScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch)  # avoid div by zero
        
        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Inverse square decay after warmup
            decay_step = step - self.warmup_steps + 1
            decay_factor = 1.0 / (decay_step ** 0.5)
            return [base_lr * decay_factor for base_lr in self.base_lrs]  
        
class GatedFF(nn.Module):
    def __init__(self, input_dim, output_dim=64, dropout=0.1, v_activation=nn.GELU(), g_activation=None, use_norm=True, use_bias=True):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.gate_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(output_dim, bias=use_bias)
        self.v_activation = v_activation
        self.g_activation = g_activation
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        v = self.value_proj(hidden_states)                 # [B, T, output_dim]
        if self.v_activation is not None:
            v = self.v_activation(v)
        g = self.gate_proj(hidden_states)
        if self.g_activation is not None:
            g = self.g_activation(g)
        x = v * g                                          # gated interaction
        if self.use_norm:
            x = self.norm(x)                               # normalize gated output
        if self.dropout: 
            x = self.dropout(x)
        return x                                           # [B, T, output_dim]