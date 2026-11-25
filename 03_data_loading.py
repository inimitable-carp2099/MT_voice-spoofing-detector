# 3/8. Data loading: protocols, audio decoding, 4 s cropping, training dataset.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 3. Data loading ========================
def load_protocol(path):
    """ASVspoof 2019 LA protocol: speaker, file, -, attack id, label."""
    cols = ["speaker", "file", "unused", "attack_id", "label"]
    df = pd.read_csv(path, sep=" ", names=cols)
    df["label_bin"] = (df["label"] == "spoof").astype(int)   # 0 = bona fide, 1 = spoof
    return df


def load_via_ffmpeg(path, sr=SAMPLE_RATE):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        raise RuntimeError("ffmpeg decoding failed")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path, sr=SAMPLE_RATE):
    """Mono waveform at 16 kHz; tries soundfile, librosa and FFmpeg in turn. Returns None on failure."""
    try:
        wav, file_sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        if len(wav) > 0:
            return wav
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wav, _ = librosa.load(path, sr=sr, mono=True)
        if len(wav) > 0:
            return wav
    except Exception:
        pass
    try:
        wav = load_via_ffmpeg(path, sr)
        if len(wav) > 0:
            return wav
    except Exception:
        pass
    return None


def pad_or_crop(wav, max_len=MAX_LEN, random_start=False):
    """Fragment of max_len samples: random start in training, first 4 s in evaluation;
    shorter recordings are repeated cyclically."""
    if len(wav) >= max_len:
        start = random.randint(0, len(wav) - max_len) if random_start else 0
        return wav[start:start + max_len]
    reps = int(np.ceil(max_len / len(wav)))
    return np.tile(wav, reps)[:max_len]


class ASVspoofDataset(Dataset):
    """ASVspoof 2019 LA train/dev; augmentation is applied only in training."""

    def __init__(self, protocol_df, flac_dir, train=True, max_len=MAX_LEN):
        self.df = protocol_df.reset_index(drop=True)
        self.flac_dir = flac_dir
        self.train = train
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _load(self, filename):
        wav, sr = sf.read(os.path.join(self.flac_dir, filename + ".flac"), dtype="float32")
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = pad_or_crop(self._load(row["file"]), self.max_len, random_start=self.train)
        if self.train:
            wav = augment(wav, self.max_len)
        return torch.tensor(wav, dtype=torch.float32), row["label_bin"], row["attack_id"]


def collate(batch):
    wavs, labels, attacks = zip(*batch)
    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long), list(attacks)
