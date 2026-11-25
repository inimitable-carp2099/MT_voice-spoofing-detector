# 2/8. Augmentations: RawBoost, telephone channel simulation, reverberation and media codecs.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 2. Data augmentation ========================
# RawBoost: port of the reference implementation (Tak et al., ICASSP 2022,
# github.com/TakHemlata/SSL_Anti-spoofing) with its default parameters.

def _rb_rand(a, b, integer=False):
    y = np.random.uniform(a, b)
    return int(y) if integer else y


def _rb_norm(x, always=False):
    m = np.max(np.abs(x)) + 1e-12
    return x / m if (always or m > 1) else x


def _rb_notch_coeffs(n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
                     min_coeff=10, max_coeff=100, min_g=0, max_g=0, fs=16000):
    """Cascade of n_bands notch FIR filters with random centre frequency and bandwidth."""
    b = np.array([1.0])
    for _ in range(n_bands):
        fc = _rb_rand(min_f, max_f)
        bw = _rb_rand(min_bw, max_bw)
        c = _rb_rand(min_coeff, max_coeff, True)
        if c % 2 == 0:
            c += 1
        f1, f2 = max(fc - bw / 2, 1e-3), min(fc + bw / 2, fs / 2 - 1e-3)
        b = np.convolve(signal.firwin(c, [f1, f2], window="hamming", fs=fs), b)
    g = _rb_rand(min_g, max_g)
    _, h = signal.freqz(b, 1, fs=fs)
    return (10 ** (g / 20)) * b / np.max(np.abs(h))


def _rb_filter(x, b):
    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n)))
    return y[int(n / 2):int(n / 2) + len(x)]


def lnl_convolutive_noise(x, sr=16000, n_f=5, min_bias=5, max_bias=20):
    """Linear and non-linear convolutive noise (RawBoost, algorithm 1)."""
    x = x.astype(np.float64)
    y = np.zeros_like(x)
    min_g = max_g = 0.0
    for i in range(n_f):
        if i == 1:
            min_g, max_g = min_g - min_bias, max_g - max_bias
        y = y + _rb_filter(np.power(x, i + 1), _rb_notch_coeffs(min_g=min_g, max_g=max_g, fs=sr))
    y = y - np.mean(y)
    return _rb_norm(y).astype(np.float32)


def isd_additive_noise(x, p=10, g_sd=2):
    """Impulsive signal-dependent additive noise (RawBoost, algorithm 2)."""
    x = x.astype(np.float64)
    y = x.copy()
    n = int(len(x) * _rb_rand(0, p) / 100)
    idx = np.random.permutation(len(x))[:n]
    f_r = (2 * np.random.rand(n) - 1) * (2 * np.random.rand(n) - 1)
    y[idx] = x[idx] + g_sd * x[idx] * f_r
    return _rb_norm(y).astype(np.float32)


def ssi_additive_noise(x, snr_min=10, snr_max=40, sr=16000):
    """Stationary signal-independent coloured noise (RawBoost, algorithm 3)."""
    x = x.astype(np.float64)
    noise = _rb_norm(_rb_filter(np.random.normal(0, 1, len(x)), _rb_notch_coeffs(fs=sr)), always=True)
    snr = _rb_rand(snr_min, snr_max)
    noise = noise / (np.linalg.norm(noise) + 1e-12) * np.linalg.norm(x) / 10.0 ** (0.05 * snr)
    return (x + noise).astype(np.float32)


def rawboost(x, sr=16000):
    """One to three RawBoost components in random order, then peak normalisation if needed."""
    fns = [lnl_convolutive_noise, isd_additive_noise, ssi_additive_noise]
    random.shuffle(fns)
    k = random.randint(1, 3)
    y = x
    for fn in fns[:k]:
        y = fn(y, sr) if fn is lnl_convolutive_noise else fn(y)
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def alaw_compand(x):
    """G.711 A-law companding with 8-bit quantisation."""
    a = 87.6
    eps = 1e-12
    x = np.clip(x, -1, 1)
    sign = np.sign(x)
    absx = np.maximum(np.abs(x), eps)
    y = np.where(absx < 1 / a,
                 a * absx / (1 + np.log(a)),
                 (1 + np.log(a * absx)) / (1 + np.log(a)))
    y_q = np.round(sign * y * 127) / 127
    absy = np.maximum(np.abs(y_q), eps)
    x_rec = np.where(absy < 1 / (1 + np.log(a)),
                     absy * (1 + np.log(a)) / a,
                     np.exp(absy * (1 + np.log(a)) - 1) / a)
    return (np.sign(y_q) * x_rec).astype(np.float32)


def mulaw_compand(x):
    """G.711 mu-law companding with 8-bit quantisation."""
    mu = 255
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y_q = np.round(y * 127) / 127
    return (np.sign(y_q) * np.expm1(np.abs(y_q) * np.log1p(mu)) / mu).astype(np.float32)


def telephone_bandwidth(x, sr=16000):
    """Band limitation to 4 kHz by resampling 16 -> 8 -> 16 kHz."""
    x_8k = librosa.resample(x, orig_sr=sr, target_sr=8000)
    x_back = librosa.resample(x_8k, orig_sr=8000, target_sr=sr)
    if len(x_back) < len(x):
        x_back = np.pad(x_back, (0, len(x) - len(x_back)))
    return x_back[:len(x)].astype(np.float32)


def lossy_codec_simulation(x, sr=16000):
    """Coarse lossy compression: quantisation to 6/8/10 bits and optional low-pass filter."""
    bits = random.choice([6, 8, 10])
    x_q = np.round(x * (2 ** (bits - 1))) / (2 ** (bits - 1))
    if random.random() < 0.5:
        cutoff = random.randint(3000, 7000)
        sos = signal.butter(4, cutoff, btype="low", fs=sr, output="sos")
        x_q = signal.sosfilt(sos, x_q)
    return x_q.astype(np.float32)


def telephone_augment(x, sr=16000):
    choice = random.choice(["alaw", "mulaw", "bandwidth", "lossy", "alaw_bw", "lossy_bw"])
    if choice == "alaw":
        return alaw_compand(x)
    if choice == "mulaw":
        return mulaw_compand(x)
    if choice == "bandwidth":
        return telephone_bandwidth(x, sr)
    if choice == "lossy":
        return lossy_codec_simulation(x, sr)
    if choice == "alaw_bw":
        return alaw_compand(telephone_bandwidth(x, sr))
    return lossy_codec_simulation(telephone_bandwidth(x, sr), sr)


def fix_length(y, n):
    """Exactly n samples, NaN and inf replaced by zeros."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n].astype(np.float32)


def reverb_augment(x, sr=16000):
    """Synthetic room reverberation: direct sound plus exponentially decaying noise, RT60 0.15-0.8 s."""
    rt60 = random.uniform(0.15, 0.8)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rir = np.random.randn(n) * np.exp(-6.9 * t / rt60)
    rir *= random.uniform(0.05, 0.4)
    rir[0] = 1.0
    y = signal.fftconvolve(x, rir, mode="full")[:len(x)]
    y *= np.sqrt(np.mean(x ** 2) + 1e-12) / np.sqrt(np.mean(y ** 2) + 1e-12)
    return fix_length(y, len(x))


MEDIA_CODECS = [("libmp3lame", "mp3", [24, 32, 48, 64]),
                ("libopus", "ogg", [12, 16, 24, 32]),
                ("aac", "adts", [24, 32, 48])]


def media_codec(x, sr=16000):
    """Encoding and decoding with MP3, Opus or AAC via FFmpeg; returns the input unchanged on failure."""
    codec, container, bitrates = random.choice(MEDIA_CODECS)
    br = random.choice(bitrates)
    try:
        raw = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        enc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
             "-c:a", codec, "-b:a", f"{br}k", "-f", container, "pipe:1"],
            input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if enc.returncode != 0 or len(enc.stdout) == 0:
            return x
        dec = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "f32le", "-acodec", "pcm_f32le",
             "-ac", "1", "-ar", str(sr), "pipe:1"],
            input=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        if dec.returncode != 0 or len(dec.stdout) == 0:
            return x
        return fix_length(np.frombuffer(dec.stdout, dtype=np.float32).copy(), len(x))
    except Exception:
        return x


def wild_augment(x, sr=16000):
    """Reverberation, media codec, or reverberation followed by a media codec."""
    choice = random.choice(["reverb", "codec", "reverb_codec"])
    if choice == "reverb":
        return reverb_augment(x, sr)
    if choice == "codec":
        return media_codec(x, sr)
    return media_codec(reverb_augment(x, sr), sr)


def augment(wav, n=MAX_LEN, sr=SAMPLE_RATE):
    """Full training pipeline applied to a 4 s fragment."""
    if random.random() < RAWBOOST_PROB:
        wav = fix_length(rawboost(wav, sr), n)
    if random.random() < TELEPHONE_PROB:
        wav = fix_length(telephone_augment(wav, sr), n)
    if random.random() < WILD_PROB:
        wav = fix_length(wild_augment(wav, sr), n)
    return wav
