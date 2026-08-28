import torch


# ==========================================================
# DATA
# ==========================================================

DATA_PATH = r"data/processed/final_genetic_features.csv"

TARGET_COLUMN = "Target"

METADATA_COLUMNS = [
    "GSM_ID",
    "Source",
    "Subtype",
    "Target"
]


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

RANDOM_STATE = 42


# ==========================================================
# MODEL
# ==========================================================

NUM_GENES = 1000

EMBEDDING_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

FEED_FORWARD_DIM = 128

DROPOUT = 0.30


# ==========================================================
# TRAINING
# ==========================================================

NUM_CLASSES = 2

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 10


# ==========================================================
# CROSS VALIDATION
# ==========================================================

NUM_FOLDS = 5


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Using device:", DEVICE)