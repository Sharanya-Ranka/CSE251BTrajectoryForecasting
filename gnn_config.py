# Model configuration
MODEL_CONFIG = {
    "D_INPUT": 9,  # Input features (position, velocity, etc.)
    "D_MODEL": 128,  # Model dimension
    "N_HEAD": 4,  # Number of attention heads
    "DROPOUT": 0.1,  # Dropout rate
    "WINDOW_SIZE": 50,  # Sequence length
    "STRIDE": 1,  # Stride for sliding window
}

# Training configuration
TRAIN_CONFIG = {
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 0.001,
    "WEIGHT_DECAY": 0.01,
    "NUM_EPOCHS": 100,
    "EARLY_STOPPING_PATIENCE": 10,
    "WARMUP_STEPS": 1000,
    "MAX_LEARNING_RATE": 0.001,
}

# Loss weights
LOSS_WEIGHTS = {
    "PHYSICAL": 1.0,  # Position loss weight
    "SMOOTHNESS": 0.1,  # Velocity and acceleration loss weight
} 