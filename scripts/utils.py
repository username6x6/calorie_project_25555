import os
import random

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

import torchmetrics
import timm
from transformers import AutoModel

from dataset import get_loaders


# Воспроизводимость

def set_seed(seed):
    """Фиксирует seed для random, numpy и torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Модель

class DishCalorieModel(nn.Module):
    """
    Двухмодальная модель: image-энкодер из timm и текстовый энкодер из transformers.
    Выходы проецируются в общее пространство, конкатенируются и подаются
    в регрессионную голову. Голова предсказывает удельную калорийность (cal/g).
    """

    def __init__(self, cfg):
        super().__init__()

        self.image_encoder = timm.create_model(
            cfg.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)

        self.text_projection = nn.Linear(
            self.text_encoder.config.hidden_size, cfg.PROJECTION_DIM
        )
        self.image_projection = nn.Linear(
            self.image_encoder.num_features, cfg.PROJECTION_DIM
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.PROJECTION_DIM * 2, cfg.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.PROJECTION_DIM, 1),
        )

        self._set_encoder_grad(cfg.UNFREEZE_IMAGE_ENCODER, cfg.UNFREEZE_TEXT_ENCODER)

    def _set_encoder_grad(self, unfreeze_image, unfreeze_text):
        for param in self.image_encoder.parameters():
            param.requires_grad = unfreeze_image
        for param in self.text_encoder.parameters():
            param.requires_grad = unfreeze_text

    def forward(self, image, input_ids, attention_mask):
        # Признаки изображения
        image_features = self.image_encoder(image)

        # Признаки текста: [CLS]-эмбеддинг
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Проекции в общее пространство
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        combined = torch.cat([image_proj, text_proj], dim=1)

        # Выход — удельная калорийность (cal/g), без активации
        return self.classifier(combined).squeeze(-1)


# Один шаг обучения и валидации

def _train_one_epoch(model, loader, optimizer, criterion, metric, device,
                     log_every, max_grad_norm):
    model.train()
    metric.reset()
    running_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        target_per_g = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Предсказание удельной калорийности, затем пересчёт в абсолютные калории
        pred_per_g = model(image, input_ids, attention_mask)
        pred_cal = pred_per_g * mass
        target_cal = target_per_g * mass

        loss = criterion(pred_cal, target_cal)

        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        running_loss += loss.item()
        metric.update(pred_cal.detach(), target_cal)

        if log_every and step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {loss.item():.2f}")

    avg_loss = running_loss / max(len(loader), 1)
    mae = metric.compute().item()
    return avg_loss, mae


@torch.no_grad()
def _validate(model, loader, criterion, metric, device):
    model.eval()
    metric.reset()
    running_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        target_per_g = batch["target"].to(device, non_blocking=True)

        pred_per_g = model(image, input_ids, attention_mask)
        pred_cal = pred_per_g * mass
        target_cal = target_per_g * mass

        loss = criterion(pred_cal, target_cal)

        running_loss += loss.item()
        metric.update(pred_cal, target_cal)

    avg_loss = running_loss / max(len(loader), 1)
    mae = metric.compute().item()
    return avg_loss, mae


# Обучение

def train(cfg):
    """
    Полный цикл обучения: seed, загрузчики, модель, оптимизатор, планировщик,
    цикл по эпохам, логирование, сохранение лучшего чекпоинта, early stopping.
    """
    set_seed(cfg.SEED)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_loader, test_loader, _ = get_loaders(cfg)
    print(f"Train батчей: {len(train_loader)}  Test батчей: {len(test_loader)}")

    model = DishCalorieModel(cfg).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Обучаемых параметров: {n_trainable:,} из {n_total:,}")

    optimizer = AdamW(
        [
            {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
            {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
            {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
            {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
            {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.NUM_EPOCHS,
        eta_min=1e-6,
    )

    criterion = nn.L1Loss()
    metric = torchmetrics.MeanAbsoluteError().to(device)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    history = {
        "train_loss": [],
        "train_mae": [],
        "test_loss": [],
        "test_mae": [],
    }
    best_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\nЭпоха {epoch}/{cfg.NUM_EPOCHS}")

        train_loss, train_mae = _train_one_epoch(
            model, train_loader, optimizer, criterion, metric, device,
            log_every=cfg.LOG_EVERY_N_STEPS,
            max_grad_norm=cfg.MAX_GRAD_NORM,
        )
        test_loss, test_mae = _validate(
            model, test_loader, criterion, metric, device,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["test_loss"].append(test_loss)
        history["test_mae"].append(test_mae)

        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"  train loss {train_loss:.2f}  train MAE {train_mae:.2f}  "
            f"| test loss {test_loss:.2f}  test MAE {test_mae:.2f}  "
            f"| lr {current_lr:.2e}"
        )

        if test_mae < best_mae:
            best_mae = test_mae
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_mae": best_mae,
                },
                cfg.MODEL_SAVE_PATH,
            )
            print(f"  чекпоинт сохранён: {cfg.MODEL_SAVE_PATH}  (MAE {best_mae:.2f})")
        else:
            epochs_without_improvement += 1
            print(f"  без улучшения {epochs_without_improvement}/{cfg.PATIENCE}")

        if epochs_without_improvement >= cfg.PATIENCE:
            print(f"\nEarly stopping: нет улучшения {cfg.PATIENCE} эпох подряд.")
            break

    print(f"\nЛучший test MAE: {best_mae:.2f}")
    return model, history