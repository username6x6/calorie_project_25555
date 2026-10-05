import os

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import AutoTokenizer


# Средние и std для нормализации, совместимые с ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size, is_train):
    """
    Пайплайн аугментаций через albumentations.
    Для train — RandomResizedCrop, отражение, поворот и изменение цвета.
    Для val/test — только ресайз и нормализация.
    """
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.85, 1.15),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DishDataset(Dataset):
    """
    Датасет блюд: изображение, токенизированный текст, реальная масса
    и удельная калорийность (cal/g) как целевая переменная.
    """

    def __init__(
        self,
        dataframe,
        tokenizer,
        transform,
        image_size,
        max_len,
        data_dir,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.image_size = image_size
        self.max_len = max_len
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, dish_id):
        """
        Открывает rgb.png. При ошибке возвращает чёрный кадр,
        чтобы обучение не падало из-за отдельных битых файлов.
        """
        img_path = os.path.join(
            self.data_dir, "images", str(dish_id), "rgb.png"
        )
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        return np.array(image)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        dish_id = row["dish_id"]
        text = str(row["ingredients_text"])

        # Реальная масса и абсолютные калории
        mass_raw = float(row["total_mass"])
        mass_raw = max(mass_raw, 1e-3)  # защита от деления на ноль
        calories = float(row["total_calories"])

        # Целевая переменная — удельная калорийность (cal/g)
        target_per_g = calories / mass_raw

        # Изображение и аугментации
        image_np = self._load_image(dish_id)
        image = self.transform(image=image_np)["image"]

        # Токенизация текста
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mass": torch.tensor(mass_raw, dtype=torch.float32),
            "target": torch.tensor(target_per_g, dtype=torch.float32),
        }


def get_loaders(cfg):
    """
    Читает подготовленный CSV, делит по split, возвращает DataLoader-ы.
    """
    csv_path = os.path.join(cfg.DATA_DIR, cfg.PREPARED_CSV)
    df = pd.read_csv(csv_path)

    df["ingredients_text"] = df["ingredients_text"].fillna("unknown")

    # Защита от нулевой массы во всём датасете — иначе cal/g уходит в бесконечность
    df["total_mass"] = df["total_mass"].clip(lower=1e-3)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train примеров: {len(train_df)}  Test примеров: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

    train_transform = build_transforms(cfg.IMAGE_SIZE, is_train=True)
    test_transform = build_transforms(cfg.IMAGE_SIZE, is_train=False)

    train_ds = DishDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        transform=train_transform,
        image_size=cfg.IMAGE_SIZE,
        max_len=cfg.MAX_LEN,
        data_dir=cfg.DATA_DIR,
    )

    test_ds = DishDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        transform=test_transform,
        image_size=cfg.IMAGE_SIZE,
        max_len=cfg.MAX_LEN,
        data_dir=cfg.DATA_DIR,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, tokenizer