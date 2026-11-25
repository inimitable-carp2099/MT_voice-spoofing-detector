# 8/8. Decision: threshold, calibration, rejection of uncertain decisions, single-file scoring.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 8. Decision procedure ========================
def dev_threshold(model):
    """Decision threshold theta: EER point on the ASVspoof 2019 LA development set."""
    df = build_test_set("dev")
    s, _ = score_test_set(model, df)
    labels = df["label_bin"].values
    _, theta = eer_and_threshold(s[labels == 0], s[labels == 1])
    return float(theta)


def calibrate_threshold(bona_scores, p):
    """Threshold from bona fide recordings of the target channel only:
    the p-th percentile gives an expected false rejection rate of p %."""
    return float(np.percentile(bona_scores, p))


def rejection_limit(values, q):
    """Value above which a decision is deferred, so that the q % least confident decisions
    of a reference set are referred for additional verification."""
    return float(np.percentile(values, 100 - q))


class SpoofDecision:
    """Accept / reject / refer to additional verification.

    criterion = "u":      evidential uncertainty u (recommended for q up to 10 %);
    criterion = "margin": distance to the threshold |s - theta| (recommended for q around 20 %).
    """

    def __init__(self, theta, criterion="u", limit=None):
        self.theta = theta
        self.criterion = criterion
        self.limit = limit

    @classmethod
    def fit(cls, theta, ref_scores, ref_uncertainty, criterion="u", q=10):
        """Sets the rejection limit on reference scores (for example, a validation set of the target channel)."""
        if criterion == "u":
            values = ref_uncertainty
        else:
            values = -np.abs(ref_scores - theta)
        return cls(theta, criterion, rejection_limit(values, q))

    def _doubt(self, s, u):
        if self.limit is None:
            return np.zeros_like(s, dtype=bool)
        values = u if self.criterion == "u" else -np.abs(s - self.theta)
        return values > self.limit

    def decide(self, s, u):
        s, u = np.asarray(s), np.asarray(u)
        decision = np.where(s >= self.theta, "bonafide", "spoof").astype(object)
        decision[self._doubt(s, u)] = "review"
        return decision


@torch.no_grad()
def score_file(model, path):
    """Score s and uncertainty u for a single recording (first 4 s, standard protocol)."""
    wav = load_audio(path)
    if wav is None:
        raise ValueError(f"cannot read {path}")
    x = torch.from_numpy(pad_or_crop(wav).astype(np.float32)).unsqueeze(0).to(DEVICE)
    out = model(x)
    s = out["spoof_score"].float().item()
    u = evidential_uncertainty(out["evidence"].float()).item()
    return s, u


# model = load_model(FINAL_CKPT_PATH)
# theta = dev_threshold(model)
# _, itw = evaluate_test_set(model, "ITW")
# val_mask, _ = itw_speaker_halves(itw["group"].values)
# decider = SpoofDecision.fit(theta, itw["score"].values[val_mask], itw["u"].values[val_mask], criterion="u", q=10)
# s, u = score_file(model, "example.flac")
# print(decider.decide([s], [u])[0])
