# 6/8. Training loop: AdamW, OneCycleLR, FP16, best/final checkpoints, resume.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# %% ======================== Cell 6. Training ========================
def train():
    train_ds = ASVspoofDataset(load_protocol(TRAIN_PROTOCOL), TRAIN_FLAC_DIR, train=True)
    dev_ds = ASVspoofDataset(load_protocol(DEV_PROTOCOL), DEV_FLAC_DIR, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS_TRAIN, collate_fn=collate, drop_last=True,
                              pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, collate_fn=collate)

    base_model = SpoofDetector().to(DEVICE)
    lora_params = [p for n, p in base_model.named_parameters() if "lora" in n and p.requires_grad]
    head_params = [p for n, p in base_model.named_parameters() if "lora" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA},
                                   {"params": head_params, "lr": LR_HEAD}],
                                  weight_decay=WEIGHT_DECAY)
    model = nn.DataParallel(base_model) if N_GPUS > 1 else base_model

    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR_LORA, LR_HEAD], total_steps=total_steps,
        pct_start=min(0.3, WARMUP_STEPS / max(total_steps, 1)))
    scaler = torch.amp.GradScaler("cuda")
    best_eer, start_epoch = 100.0, 0

    # Resume after an interrupted session
    if os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_eer = ckpt["epoch"] + 1, ckpt["best_eer"]

    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable / 1e6:.2f}M")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (wav, labels, _) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
            wav, labels = wav.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda"):
                out = model(wav, labels)
                loss = (out["oc_loss"].mean()
                        + CONTRASTIVE_LOSS_WEIGHT * supcon_loss(out["contrastive_emb"], labels)
                        + EVIDENTIAL_LOSS_WEIGHT * evidential_loss(out["evidence"], labels))
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        dev_eer, _ = evaluate_loader(model, dev_loader)
        print(f"epoch {epoch}: train loss {running_loss / len(train_loader):.4f}, dev EER {dev_eer:.2f}%")

        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(base_model.state_dict(), BEST_CKPT_PATH)
        torch.save({"epoch": epoch, "model": base_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "best_eer": best_eer}, RESUME_PATH)

    torch.save(base_model.state_dict(), FINAL_CKPT_PATH)
    print(f"Training finished: best dev EER {best_eer:.2f}%")


# train()
