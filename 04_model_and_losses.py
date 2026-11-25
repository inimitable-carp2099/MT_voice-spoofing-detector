# 4/8. Detector: XLS-R 300M + LoRA, layer weighting, master-token pooling, OC-Softmax, SupCon, evidential head.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 4. Detector architecture and loss functions ========================
class MasterTokenPooling(nn.Module):
    """One self-attention layer over frames with two learnable master tokens;
    the utterance embedding is the concatenation of the transformed master tokens."""

    def __init__(self, dim, n_heads=4, n_master=2):
        super().__init__()
        self.n_master = n_master
        self.master_tokens = nn.Parameter(torch.randn(n_master, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):                                       # x: [B, T, D]
        b = x.size(0)
        master = self.master_tokens.unsqueeze(0).expand(b, -1, -1)
        seq = torch.cat([master, x], dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)
        seq = self.norm2(seq + self.ffn(seq))
        return seq[:, :self.n_master, :].reshape(b, -1)         # [B, n_master * D]


class OCSoftmax(nn.Module):
    """One-class softmax loss; the score is the cosine similarity to the bona fide centre
    (higher means more bona fide)."""

    def __init__(self, dim, alpha=OC_ALPHA, m_real=OC_M_REAL, m_fake=OC_M_FAKE):
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.alpha = alpha
        self.m_real = m_real
        self.m_fake = m_fake

    def score(self, emb):
        emb = F.normalize(emb, dim=-1)
        w = F.normalize(self.center, dim=-1)
        return (emb @ w.t()).max(dim=1).values

    def forward(self, emb, labels):                             # labels: 0 = bona fide, 1 = spoof
        cos = self.score(emb)
        margins = torch.where(labels == 0, self.m_real - cos, cos - self.m_fake)
        loss = F.softplus(self.alpha * margins).mean()
        return loss, cos.detach()


class SpoofDetector(nn.Module):
    def __init__(self, wav2vec_path=WAV2VEC_MODEL_PATH, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                 layer_subset=LAYER_SUBSET):
        super().__init__()
        # Frozen XLS-R 300M with LoRA adapters in all linear layers of the transformer blocks
        self.ssl = Wav2Vec2Model.from_pretrained(wav2vec_path)
        self.ssl.config.output_hidden_states = True
        for p in self.ssl.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=LORA_DROPOUT,
                              target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                                              "intermediate_dense", "output_dense"],
                              bias="none")
        self.ssl = get_peft_model(self.ssl, lora_cfg)

        # Learnable weights of the 25 hidden-state levels, initialised with a prior on layer_subset
        n_levels = self.ssl.config.num_hidden_layers + 1
        init_logits = torch.full((n_levels,), -2.0)
        if layer_subset is not None:
            for idx in layer_subset:
                init_logits[idx] = 2.0
        self.layer_weights = nn.Parameter(init_logits)

        ssl_dim = self.ssl.config.hidden_size
        self.input_proj = nn.Linear(ssl_dim, PROJ_DIM)
        self.graph = MasterTokenPooling(PROJ_DIM, n_heads=4, n_master=2)
        emb_dim = PROJ_DIM * 2

        self.oc_softmax = OCSoftmax(emb_dim)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 128))
        self.evidential_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.GELU(),
            nn.Linear(emb_dim // 2, 2), nn.Softplus())

    def encode(self, wav):
        wav = wav.float()
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        out = self.ssl(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)              # [L, B, T, D]
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        feat = (hs * w).sum(dim=0)                              # [B, T, D]
        return self.graph(self.input_proj(feat))

    def forward(self, wav, labels=None):
        emb = self.encode(wav)
        out = {"embedding": emb}
        if labels is not None:
            out["oc_loss"], out["spoof_score"] = self.oc_softmax(emb, labels)
        else:
            out["spoof_score"] = self.oc_softmax.score(emb)
        out["contrastive_emb"] = F.normalize(self.contrastive_proj(emb), dim=-1)
        out["evidence"] = self.evidential_head(emb)
        return out


def evidential_loss(evidence, labels, n_classes=2):
    """Expected squared error under the Dirichlet distribution (Sensoy et al., 2018)."""
    evidence = evidence.float()
    alpha = evidence + 1.0
    s = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(labels, n_classes).float()
    err = ((y - alpha / s) ** 2).sum(dim=1)
    var = (alpha * (s - alpha) / (s * s * (s + 1))).sum(dim=1)
    return (err + var).mean()


def supcon_loss(emb, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al., 2020)."""
    emb = emb.float()
    sim = emb @ emb.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask_pos.fill_diagonal_(0)
    exp_sim = torch.exp(sim) * (1 - torch.eye(sim.size(0), device=sim.device))
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    denom = mask_pos.sum(dim=1).clamp(min=1)
    return (-(mask_pos * log_prob).sum(dim=1) / denom).mean()


def evidential_uncertainty(evidence):
    """u = K / S for K = 2 classes."""
    return 2.0 / (evidence + 1.0).sum(dim=1)
