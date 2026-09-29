"""Paths and hyper-parameters. Both models share every setting except the
presence of the attention layer, so the comparison isolates attention."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

HF_DATASET = "habtew/english-amharic-translation"
SEED = 42

# ---- preprocessing -------------------------------------------------------
VAL_FRAC = 0.05
TEST_FRAC = 0.05
MAX_TRAIN_TOKENS = 50      # drop training pairs longer than this (subword tokens)
MAX_EVAL_TOKENS = 100      # val/test keep long sentences for long-sentence analysis
MAX_LEN_RATIO = 3.0        # drop pairs whose token-length ratio is implausible
SPM_VOCAB_EN = 8000
SPM_VOCAB_AM = 8000

PAD, UNK, BOS, EOS = 0, 1, 2, 3

# ---- model ---------------------------------------------------------------
EMB_DIM = 256
HIDDEN = 512               # decoder hidden; encoder is bi-LSTM with HIDDEN//2 per direction
LAYERS = 2
DROPOUT = 0.3

# ---- training ------------------------------------------------------------
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
PATIENCE = 3               # early stopping on validation loss
CLIP = 1.0
MAX_DECODE_LEN = 120

# ---- translation scope (warnings shown by the apps) ------------------------
RARE_WORD_COUNT = 30       # a word seen fewer times than this in training is likely mistranslated
MIN_INPUT_WORDS = 3        # the models were trained on full sentences (16 words on average)
MAX_INPUT_WORDS = 50       # training sentences were capped at 50 subwords
