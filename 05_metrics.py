# 5/8. Metrics: EER and threshold, per-attack EER, dev-set evaluation.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 5. Metrics ========================
def compute_eer(bona_scores, spoof_scores):
    """Equal error rate in %; a higher score means more bona fide."""
    eer, _ = eer_and_threshold(bona_scores, spoof_scores)
    return eer


def eer_and_threshold(bona_scores, spoof_scores):
    """EER in % and the decision threshold at the EER point."""
    scores = np.concatenate([bona_scores, spoof_scores])
    labels = np.concatenate([np.zeros(len(bona_scores)), np.ones(len(spoof_scores))])
    order = np.argsort(scores)
    s_sorted, l_sorted = scores[order], labels[order]
    p_miss = np.cumsum(l_sorted == 0) / len(bona_scores)
    p_fa = 1 - np.cumsum(l_sorted == 1) / len(spoof_scores)
    i = np.nanargmin(np.abs(p_miss - p_fa))
    return (p_miss[i] + p_fa[i]) / 2 * 100, s_sorted[i]


def eer_per_group(scores, labels, groups):
    """EER for each spoofing attack (or other group): all bona fide trials vs the spoofed trials of the group."""
    bona = scores[labels == 0]
    return {g: round(compute_eer(bona, scores[(groups == g) & (labels == 1)]), 2)
            for g in sorted(set(groups[labels == 1]))}


@torch.no_grad()
def evaluate_loader(model, loader):
    """Pooled and per-attack EER on a labelled loader (used for the dev set during training)."""
    model.eval()
    scores, labels, attacks = [], [], []
    for wav, lab, att in loader:
        out = model(wav.to(DEVICE))
        scores.append(out["spoof_score"].float().cpu().numpy())
        labels.append(lab.numpy())
        attacks.extend(att)
    scores, labels, attacks = np.concatenate(scores), np.concatenate(labels), np.array(attacks)
    return compute_eer(scores[labels == 0], scores[labels == 1]), eer_per_group(scores, labels, attacks)
