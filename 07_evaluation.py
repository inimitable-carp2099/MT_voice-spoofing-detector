# 7/8. Evaluation: test sets (ASVspoof 2019 LA, 2021 LA, In-the-Wild, ASVspoof 5), checkpoint selection.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 7. Evaluation ========================
def load_model(ckpt_path):
    model = SpoofDetector().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return nn.DataParallel(model) if N_GPUS > 1 else model


def build_test_set(name):
    """Data frame with columns path, label_bin, group (attack id or speaker) and, for ASVspoof 5, codec."""
    if name in ("dev", "2019"):
        proto, d = (DEV_PROTOCOL, DEV_FLAC_DIR) if name == "dev" else (EVAL_PROTOCOL, EVAL_FLAC_DIR)
        p = load_protocol(proto)
        return pd.DataFrame({"path": [os.path.join(d, f + ".flac") for f in p["file"]],
                             "label_bin": p["label_bin"].values,
                             "group": p["attack_id"].astype(str).values})
    if name == "2021":
        raw = pd.read_csv(KEY_FILE_2021, sep=r"\s+", header=None, engine="python")
        raw = raw[raw[7] == "eval"]
        return pd.DataFrame({"path": [os.path.join(FLAC_2021, f + ".flac") for f in raw[1]],
                             "label_bin": (raw[5] == "spoof").astype(int).values,
                             "group": raw[4].astype(str).values}).reset_index(drop=True)
    if name == "ITW":
        itw = pd.read_csv(ITW_META)
        return pd.DataFrame({"path": [os.path.join(ITW_ROOT, f) for f in itw["file"]],
                             "label_bin": (itw["label"] == "spoof").astype(int).values,
                             "group": itw["speaker"].astype(str).values})
    if name == "ASV5":
        tsv = os.path.join(ASV5_PROTO_DIR, "ASVspoof5.eval.track_1.tsv")
        folders = (sorted(glob.glob(os.path.join(ASV5_AUDIO_DIR, "flac_E_*", "flac_E_eval")))
                   or sorted(glob.glob(os.path.join(ASV5_AUDIO_DIR, "flac_E_*"))))
        file_map = {e.name[:-5]: e.path for f in folders for e in os.scandir(f) if e.name.endswith(".flac")}
        cols = ["speaker", "file", "gender", "codec", "num", "src_spk", "cond", "attack_id", "label", "unused"]
        raw = pd.read_csv(tsv, sep=r"\s+", names=cols, engine="python")
        df = pd.DataFrame({"path": raw["file"].map(file_map),
                           "label_bin": (raw["label"] == "spoof").astype(int).values,
                           "group": raw["attack_id"].astype(str).values,
                           "codec": raw["codec"].astype(str).values})
        return df.dropna(subset=["path"]).reset_index(drop=True)
    raise ValueError(name)


class EvalDataset(Dataset):
    """Standard protocol: the first 4 s of each recording, no silence trimming."""

    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = load_audio(row["path"])
        if wav is None:
            wav = np.zeros(self.max_len, dtype=np.float32)
        return torch.from_numpy(pad_or_crop(wav, self.max_len).astype(np.float32)), int(row["label_bin"])


@torch.no_grad()
def score_test_set(model, df, batch_size=128, num_workers=4):
    """Authenticity score s and evidential uncertainty u for every recording."""
    loader = DataLoader(EvalDataset(df), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    s_all, u_all = [], []
    for wav, _ in tqdm(loader, desc="scoring"):
        with torch.amp.autocast("cuda"):
            out = model(wav.to(DEVICE, non_blocking=True))
        s_all.append(out["spoof_score"].float().cpu().numpy())
        u_all.append(evidential_uncertainty(out["evidence"].float()).cpu().numpy())
    return np.concatenate(s_all), np.concatenate(u_all)


def evaluate_test_set(model, name):
    """Pooled EER, EER per attack (ASVspoof) or per codec (ASVspoof 5); scores are saved to CSV."""
    df = build_test_set(name)
    s, u = score_test_set(model, df)
    labels, groups = df["label_bin"].values, df["group"].values
    result = {"set": name, "EER": round(compute_eer(s[labels == 0], s[labels == 1]), 2)}
    if name != "ITW":
        result["per_attack"] = eer_per_group(s, labels, groups)
    if name == "ASV5":
        codecs = df["codec"].values
        result["per_codec"] = {c: round(compute_eer(s[(codecs == c) & (labels == 0)],
                                                    s[(codecs == c) & (labels == 1)]), 2)
                               for c in sorted(set(codecs))}
    out = df.assign(score=s, u=u)
    out.to_csv(f"{OUT_DIR}/scores_{RUN_NAME}_{name}.csv", index=False)
    return result, out


def itw_speaker_halves(groups):
    """Speaker-disjoint halves of In-the-Wild: speakers at even positions of the sorted list
    form the validation half used for checkpoint selection."""
    speakers = sorted(set(groups))
    return np.isin(groups, speakers[::2]), np.isin(groups, speakers[1::2])


def select_checkpoint(ckpt_paths):
    """Chooses the checkpoint with the lowest EER on the validation half of In-the-Wild."""
    df = build_test_set("ITW")
    val_mask, _ = itw_speaker_halves(df["group"].values)
    labels = df["label_bin"].values
    eers = {}
    for path in ckpt_paths:
        s, _ = score_test_set(load_model(path), df)
        eers[path] = compute_eer(s[val_mask & (labels == 0)], s[val_mask & (labels == 1)])
    return min(eers, key=eers.get), eers


# best_ckpt, val_eers = select_checkpoint([BEST_CKPT_PATH, FINAL_CKPT_PATH])
# model = load_model(best_ckpt)
# for name in ("2019", "2021", "ITW", "ASV5"):
#     result, _ = evaluate_test_set(model, name)
#     print(name, result["EER"])
