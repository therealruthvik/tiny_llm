"""
tiny_llm.py — Build a GPT-style LLM from scratch in PyTorch
=============================================================
Run each section in order. Every component matches what was shown
in the interactive explainer.

Requirements:
    pip install torch

Optional (for nicer training curves):
    pip install matplotlib

Architecture:
    - Character-level tokenizer  (no external libs needed)
    - Token + Positional embeddings
    - N × Transformer blocks  (Pre-LN, causal self-attention + FFN)
    - Language-model head  (linear → softmax)
    - Autoregressive inference with temperature sampling
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 0.  HYPER-PARAMETERS
#     Tweak these to make the model larger or smaller.
# ──────────────────────────────────────────────────────────────────────────────

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Model shape
D_MODEL   = 128       # embedding / hidden size (GPT-2 Small uses 768)
N_HEADS   = 4         # attention heads  (D_MODEL must be divisible by N_HEADS)
N_LAYERS  = 4         # transformer blocks stacked
FF_DIM    = D_MODEL * 4   # feed-forward inner dim (standard 4× rule)
MAX_SEQ   = 128       # maximum context length (tokens)
DROPOUT   = 0.1

# Training
BATCH     = 32
LR        = 3e-4
STEPS     = 5000      # training iterations (raise to 10k+ for better results)
EVAL_EVERY = 500      # print loss every N steps

print(f"Device: {DEVICE}")
print(f"d_model={D_MODEL}  heads={N_HEADS}  layers={N_LAYERS}  ff_dim={FF_DIM}")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  TOKENIZER — character-level BPE substitute
#     Maps every unique character to an integer ID and back.
#     Real models use SentencePiece / tiktoken for subword BPE.
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Simplest possible tokenizer: one token per character.
    vocab size = number of unique chars in the training corpus.
    """
    def __init__(self, text: str):
        chars       = sorted(set(text))          # unique characters
        self.vocab  = chars
        self.s2i    = {c: i for i, c in enumerate(chars)}  # char → id
        self.i2s    = {i: c for i, c in enumerate(chars)}  # id  → char
        self.vocab_size = len(chars)
        print(f"\n[Tokenizer] vocab_size={self.vocab_size}")
        print(f"  First 20 tokens: {chars[:20]}")

    def encode(self, text: str) -> list[int]:
        return [self.s2i[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.i2s[i] for i in ids)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATASET — sliding-window chunks of the training text
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(token_ids: list[int], seq_len: int):
    """
    Turn a flat list of token IDs into (input, target) pairs.
    Input:  tokens[i : i+seq_len]
    Target: tokens[i+1 : i+seq_len+1]  (predict each next token)
    """
    data = torch.tensor(token_ids, dtype=torch.long)
    xs, ys = [], []
    for i in range(0, len(data) - seq_len - 1, seq_len // 2):  # 50% overlap
        xs.append(data[i : i + seq_len])
        ys.append(data[i + 1 : i + seq_len + 1])
    X = torch.stack(xs)   # (N, seq_len)
    Y = torch.stack(ys)   # (N, seq_len)
    # train / val split 90/10
    split = int(0.9 * len(X))
    return (X[:split], Y[:split]), (X[split:], Y[split:])


def get_batch(X, Y, batch_size: int):
    """Sample a random mini-batch."""
    idx = torch.randint(len(X), (batch_size,))
    return X[idx].to(DEVICE), Y[idx].to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  POSITIONAL ENCODING
#     Adds a unique signal for each position so the model knows token order.
#     We use sinusoidal encoding (original "Attention Is All You Need").
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    This creates a unique fingerprint for every position that:
      - is deterministic (no extra parameters to learn)
      - has the same magnitude as embeddings
      - generalises to lengths unseen during training
    """
    def __init__(self, d_model: int, max_seq: int):
        super().__init__()
        pe = torch.zeros(max_seq, d_model)                   # (T, D)
        pos = torch.arange(max_seq).unsqueeze(1)             # (T, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000) / d_model)
        )                                                     # (D/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, T, D)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MULTI-HEAD CAUSAL SELF-ATTENTION
#     The core of the transformer. Every token attends to all previous tokens.
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadCausalSelfAttention(nn.Module):
    """
    Scaled dot-product attention with causal (triangular) mask.

    Steps per forward pass:
      1. Project input x → Q, K, V  (separate learned linear layers)
      2. Split into N_HEADS independent heads
      3. Compute attention scores = Q @ Kᵀ / sqrt(d_k)
      4. Apply causal mask (zero out future positions)
      5. Softmax → attention weights
      6. Weighted sum of V → attended output
      7. Concatenate heads, project back to d_model
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k     = d_model // n_heads  # dimension per head

        # Single combined projection for Q, K, V (efficient)
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj   = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape

        # 1. Compute Q, K, V in one matmul, then split
        qkv = self.qkv_proj(x)                          # (B, T, 3D)
        Q, K, V = qkv.split(D, dim=-1)                  # each (B, T, D)

        # 2. Split into heads: (B, T, D) → (B, n_heads, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # 3. Scaled dot-product scores
        scale  = math.sqrt(self.d_k)
        scores = (Q @ K.transpose(-2, -1)) / scale      # (B, n_heads, T, T)

        # 4. Causal mask: token i cannot attend to token j > i
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax over the key dimension
        weights = F.softmax(scores, dim=-1)              # (B, n_heads, T, T)
        weights = self.attn_drop(weights)

        # 6. Weighted sum of values
        out = weights @ V                                # (B, n_heads, T, d_k)

        # 7. Merge heads back: (B, n_heads, T, d_k) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.out_proj(out))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FEED-FORWARD NETWORK (FFN)
#     Applied independently to each token position after attention.
#     Stores factual / pattern knowledge from training.
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation between them.

    Structure:
        Linear(d_model → ff_dim)  →  GELU  →  Linear(ff_dim → d_model)

    GELU (Gaussian Error Linear Unit):
        GELU(x) ≈ x * Φ(x)  where Φ is the Gaussian CDF
        Smoother than ReLU; used in BERT, GPT-2, GPT-3.
    """
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)   # (B, T, D) → (B, T, D)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  LAYER NORMALIZATION
#     Stabilises training by normalising across the feature dimension.
#     We use Pre-LN (norm before sublayer) — more stable than original Post-LN.
# ──────────────────────────────────────────────────────────────────────────────

# nn.LayerNorm is built into PyTorch, so we use it directly.
# Internally it computes:
#   ŷ = (x - mean(x)) / sqrt(var(x) + ε)  ×  γ  +  β
# where γ (weight) and β (bias) are learned per-feature parameters.


# ──────────────────────────────────────────────────────────────────────────────
# 7.  TRANSFORMER BLOCK
#     One complete layer: Pre-LN → Attention → residual
#                        + Pre-LN → FFN      → residual
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (GPT-2 style).

    Diagram:
        x ──┬──► LayerNorm ──► Attention ──► + ──┬──► LayerNorm ──► FFN ──► + ──►
            │                               ▲    │                           ▲
            └───────────────────────────────┘    └───────────────────────────┘
                       residual connection                residual connection
    """
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x):
        # Residual connections wrap every sublayer
        x = x + self.attn(self.ln1(x))   # attention sublayer
        x = x + self.ffn(self.ln2(x))    # ffn sublayer
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 8.  FULL GPT-STYLE LANGUAGE MODEL
#     Token embedding → positional encoding → N blocks → LM head
# ──────────────────────────────────────────────────────────────────────────────

class TinyLLM(nn.Module):
    """
    Full autoregressive language model.

    Forward pass returns logits of shape (B, T, vocab_size).
    Loss is cross-entropy between predicted and actual next tokens.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ff_dim,
                 max_seq, dropout):
        super().__init__()

        # Embedding table: vocab_size × d_model learnable parameters
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding (sinusoidal, no extra params)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq)

        self.drop = nn.Dropout(dropout)

        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (Pre-LN style needs one at the end too)
        self.ln_f = nn.LayerNorm(d_model)

        # Language model head: project d_model → vocab_size logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights
        # (saves params; common in GPT-2, BERT)
        self.lm_head.weight = self.tok_emb.weight

        # Initialise weights (important for stable training)
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n[TinyLLM] Parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) next-token ids (optional, for computing loss)
        returns: logits (B, T, V), loss (scalar or None)
        """
        B, T = idx.shape

        # 1. Token embeddings: integer IDs → dense vectors
        x = self.tok_emb(idx)           # (B, T, D)

        # 2. Add positional encoding
        x = self.pos_enc(x)             # (B, T, D)
        x = self.drop(x)

        # 3. Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)                # (B, T, D)

        # 4. Final layer norm
        x = self.ln_f(x)               # (B, T, D)

        # 5. Project to vocabulary logits
        logits = self.lm_head(x)       # (B, T, vocab_size)

        # 6. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten (B, T) into (B*T,) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, V)
                targets.view(-1)                    # (B*T,)
            )

        return logits, loss


# ──────────────────────────────────────────────────────────────────────────────
# 9.  INFERENCE — autoregressive text generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 40) -> str:
    """
    Generate text autoregressively from a prompt.

    Algorithm:
        1. Encode prompt → token ids
        2. Forward pass → logits for the last position
        3. Apply temperature scaling  (higher T = more random)
        4. Optional top-k filtering   (keep only top-k candidates)
        5. Softmax → probabilities
        6. Sample one token
        7. Append to sequence; go to step 2
    """
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # Crop to MAX_SEQ if the context is too long
        x_cond = x[:, -MAX_SEQ:]

        # Forward pass — we only care about the LAST token's logits
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]          # (1, vocab_size)

        # Temperature: divide logits before softmax
        #   T < 1  → sharper distribution (more deterministic)
        #   T > 1  → flatter distribution (more random)
        logits = logits / temperature

        # Top-k filtering: zero out all but the top-k logits
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = float("-inf")

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


# ──────────────────────────────────────────────────────────────────────────────
# 10. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(model, train_data, val_data, steps: int):
    """
    Simple AdamW training loop with periodic validation loss reporting.

    AdamW = Adam with decoupled weight decay (Loshchilov & Hutter 2019).
    Learning rate warmup is omitted here for simplicity.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    model.train()

    train_X, train_Y = train_data
    val_X,   val_Y   = val_data

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        # Sample a batch
        xb, yb = get_batch(train_X, train_Y, BATCH)

        # Forward + loss
        _, loss = model(xb, yb)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())

        if step % EVAL_EVERY == 0 or step == 1:
            # Validation loss (no gradient)
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_X, val_Y, BATCH * 2)
                _, val_loss = model(xv, yv)
            model.train()

            avg_train = sum(losses[-EVAL_EVERY:]) / len(losses[-EVAL_EVERY:])
            elapsed   = time.time() - t0
            print(f"  step {step:5d}/{steps}  "
                  f"train_loss={avg_train:.4f}  "
                  f"val_loss={val_loss.item():.4f}  "
                  f"elapsed={elapsed:.1f}s")

    return losses


# ──────────────────────────────────────────────────────────────────────────────
# 11. PUT IT ALL TOGETHER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Training corpus ──────────────────────────────────────────────────────
    # Using a short built-in string so the script runs with zero downloads.
    # Replace CORPUS with open("your_file.txt").read() for a real dataset.
    # Good free datasets: Shakespeare (tinyshakespeare), Project Gutenberg books,
    # or the WikiText-2 dataset.
    CORPUS = """
To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heartache and the thousand natural shocks that flesh is heir to tis a consummation
devoutly to be wished to die to sleep to sleep perchance to dream aye there is the rub
for in that sleep of death what dreams may come when we have shuffled off this mortal coil
must give us pause there is the respect that makes calamity of so long life
for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely
the pangs of despised love the laws delay the insolence of office and the spurns
that patient merit of the unworthy takes when he himself might his quietus make
with a bare bodkin who would fardels bear to grunt and sweat under a weary life
but that the dread of something after death the undiscovered country from whose bourn
no traveler returns puzzles the will and makes us rather bear those ills we have
than fly to others that we know not of thus conscience does make cowards of us all
and thus the native hue of resolution is sicklied over with the pale cast of thought
and enterprises of great pitch and moment with this regard their currents turn awry
and lose the name of action soft you now the fair ophelia nymph in thy orisons
be all my sins remembered
""" * 30  # repeat to give the model more data

    print("=" * 60)
    print("STEP 1: TOKENIZE")
    print("=" * 60)
    tokenizer = CharTokenizer(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    print(f"  Corpus: {len(CORPUS):,} chars  →  {len(token_ids):,} tokens")

    print("\n" + "=" * 60)
    print("STEP 2: BUILD DATASET")
    print("=" * 60)
    (train_X, train_Y), (val_X, val_Y) = build_dataset(token_ids, MAX_SEQ)
    print(f"  Train: {train_X.shape}   Val: {val_X.shape}")
    print(f"  Example input:  {tokenizer.decode(train_X[0, :20].tolist())!r}")
    print(f"  Example target: {tokenizer.decode(train_Y[0, :20].tolist())!r}")

    print("\n" + "=" * 60)
    print("STEP 3: BUILD MODEL")
    print("=" * 60)
    model = TinyLLM(
        vocab_size = tokenizer.vocab_size,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        ff_dim     = FF_DIM,
        max_seq    = MAX_SEQ,
        dropout    = DROPOUT,
    ).to(DEVICE)

    # Print a summary of each layer's shape
    print("\n  Layer breakdown:")
    for name, param in model.named_parameters():
        print(f"    {name:40s}  {list(param.shape)}")

    print("\n" + "=" * 60)
    print("STEP 4: TRAIN")
    print("=" * 60)
    losses = train(model, (train_X, train_Y), (val_X, val_Y), STEPS)

    # Optional: plot training curve
    try:
        import matplotlib.pyplot as plt
        window = 50
        smooth = [sum(losses[max(0,i-window):i+1])/min(i+1,window)
                  for i in range(len(losses))]
        plt.figure(figsize=(10, 4))
        plt.plot(losses, alpha=0.2, color="steelblue", label="raw")
        plt.plot(smooth, color="steelblue", label=f"{window}-step avg")
        plt.xlabel("Step")
        plt.ylabel("Cross-entropy loss")
        plt.title("Training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=120)
        print("\n  Training curve saved to training_loss.png")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("STEP 5: GENERATE TEXT (INFERENCE)")
    print("=" * 60)

    prompts = [
        "To be or not",
        "the fair ophelia",
        "sleep perchance",
    ]
    for prompt in prompts:
        print(f"\n  Prompt: {prompt!r}")
        out = generate(model, tokenizer, prompt, max_new_tokens=150,
                       temperature=0.8, top_k=40)
        print(f"  Output: {out!r}")
        print("-" * 50)

    # Save the model
    torch.save({
        "model_state": model.state_dict(),
        "vocab":       tokenizer.vocab,
        "config": dict(
            vocab_size = tokenizer.vocab_size,
            d_model    = D_MODEL,
            n_heads    = N_HEADS,
            n_layers   = N_LAYERS,
            ff_dim     = FF_DIM,
            max_seq    = MAX_SEQ,
            dropout    = DROPOUT,
        )
    }, "tiny_llm.pt")
    print("\nModel saved to tiny_llm.pt")

    # ── How to load and reuse ──────────────────────────────────────────────
    # ckpt = torch.load("tiny_llm.pt")
    # tokenizer = CharTokenizer.__new__(CharTokenizer)
    # tokenizer.vocab      = ckpt["vocab"]
    # tokenizer.s2i        = {c:i for i,c in enumerate(ckpt["vocab"])}
    # tokenizer.i2s        = {i:c for i,c in enumerate(ckpt["vocab"])}
    # tokenizer.vocab_size = len(ckpt["vocab"])
    # model = TinyLLM(**ckpt["config"]).to(DEVICE)
    # model.load_state_dict(ckpt["model_state"])
    # model.eval()