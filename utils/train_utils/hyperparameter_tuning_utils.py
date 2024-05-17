import pytorch_lightning as pl

def objective(trial):
    # 하이퍼파라미터 설정
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    # 데이터 로더 업데이트
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # PyTorch Lightning Trainer 설정
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        precision=16,  # Mixed Precision Training
        accumulate_grad_batches=2  # Gradient Accumulation
    )

    # 모델 초기화 및 훈련
    pl_model = AudioSeparationModel(model)
    trainer.fit(pl_model, train_loader, val_loader)

    # 검증 손실 반환
    return trainer.callback_metrics["val_loss"].item()


# Optuna 스터디 설정 및 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print(study.best_params)
