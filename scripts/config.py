from dataclasses import dataclass, field


@dataclass
class Config:

    # Корень с данными: ожидается, что внутри data/ лежат images/, dish.csv, ingredients.csv
    DATA_DIR: str = "data"

    # Имя подготовленного CSV после EDA и очистки ингредиентов
    PREPARED_CSV: str = "dish_prepared.csv"

    # Куда сохранять чекпоинты модели
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_SAVE_PATH: str = "checkpoints/best_model.pth"

    # Сторона квадратного ресайза изображений. 224 — база для большинства CNN и ViT
    IMAGE_SIZE: int = 224

    # Максимальная длина текста в токенах после токенизации ingredients_text
    MAX_LEN: int = 64

    # Число воркеров DataLoader. В Colab 2 — безопасное значение
    NUM_WORKERS: int = 2

    # Предобученный текстовый энкодер из transformers
    TEXT_MODEL_NAME: str = "distilbert-base-uncased"

    # Имя image-энкодера из timm
    IMAGE_MODEL_NAME: str = "resnet18"

    # Размерность эмбеддинга текстового энкодера (у distilbert — 768)
    TEXT_EMBED_DIM: int = 768

    # Размерность признаков image-энкодера (у resnet18 — 512)
    IMAGE_EMBED_DIM: int = 512

    # Размер общего пространства, куда проецируются обе модальности
    PROJECTION_DIM: int = 256

    # Dropout перед финальным регрессионным слоем
    DROPOUT: float = 0.2

    # Обучение
    SEED: int = 42

    # Batch size 4
    BATCH_SIZE: int = 4

    NUM_EPOCHS: int = 25

    PATIENCE: int = 5

    # Learning rate для текстового энкодера
    TEXT_LR: float = 2e-5

    # Learning rate для image-энкодера
    IMAGE_LR: float = 1e-5

    # Learning rate для проекций и финального классификатора
    CLASSIFIER_LR: float = 1e-3

    # Weight decay для AdamW
    WEIGHT_DECAY: float = 1e-2

    # Разморозка энкодеров. Сделала False из-за переобучения
    UNFREEZE_TEXT_ENCODER: bool = True
    UNFREEZE_IMAGE_ENCODER: bool = False

    # Сохранять ли модель при улучшении MAE на val/test
    SAVE_BEST_ONLY: bool = True

    # Логировать метрики каждые N шагов обучения
    LOG_EVERY_N_STEPS: int = 50

    DEVICE: str = "cuda"

    MAX_GRAD_NORM: float = 1.0