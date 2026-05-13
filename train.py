"""
FaceMask Detection System — Model Training Pipeline
=====================================================
Transfer learning with MobileNetV2 on ImageNet weights.
Trains a 3-class classifier: with_mask / without_mask / mask_incorrect

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch-size 64 --lr 0.0001
    python src/train.py --config configs/train_config.yaml
    python src/train.py --fine-tune  # unfreeze top layers for fine-tuning
"""

import os
import sys
import argparse
import yaml
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    AveragePooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'dataset_path':    'dataset',
    'model_save_path': 'models/facemask_model.h5',
    'log_dir':         'logs/training',
    'input_size':      [224, 224],
    'num_classes':     3,
    'classes':         ['with_mask', 'without_mask', 'mask_weared_incorrect'],

    'training': {
        'epochs':                  30,
        'batch_size':              32,
        'learning_rate':           1e-4,
        'fine_tune_learning_rate': 1e-5,
        'validation_split':        0.2,
        'test_split':              0.1,
        'fine_tune_at':            100,
        'fine_tune_epochs':        10,
    },

    'augmentation': {
        'rotation_range':      20,
        'zoom_range':          0.15,
        'width_shift_range':   0.2,
        'height_shift_range':  0.2,
        'shear_range':         0.15,
        'horizontal_flip':     True,
        'brightness_range':    [0.8, 1.2],
        'fill_mode':           'nearest',
    },

    'model': {
        'base':              'MobileNetV2',
        'weights':           'imagenet',
        'include_top':       False,
        'pooling':           None,
        'dropout_1':         0.5,
        'dropout_2':         0.3,
        'dense_units_1':     128,
        'dense_units_2':     64,
    },

    'callbacks': {
        'early_stopping_patience':  7,
        'reduce_lr_patience':       4,
        'reduce_lr_factor':         0.2,
        'reduce_lr_min':            1e-7,
    }
}


def load_config(config_path: str = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            override = yaml.safe_load(f)
        # Deep merge
        for key, val in override.items():
            if isinstance(val, dict) and key in config:
                config[key].update(val)
            else:
                config[key] = val
    return config


# ─── DATA PIPELINE ────────────────────────────────────────────────────────────

def build_data_generators(config: dict):
    """
    Build train/validation/test ImageDataGenerators with augmentation.

    Returns train_gen, val_gen, test_gen
    """
    dataset_path = config['dataset_path']
    input_size   = tuple(config['input_size'])
    batch_size   = config['training']['batch_size']
    aug_cfg      = config['augmentation']

    # Verify dataset structure
    classes = config['classes']
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            raise FileNotFoundError(
                f"Missing class folder: {cls_path}\n"
                f"Run: python scripts/prepare_dataset.py"
            )
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
        print(f"  📁  {cls}: {count} images")

    # Augmented generator for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=config['training']['validation_split'],
        rotation_range=aug_cfg['rotation_range'],
        zoom_range=aug_cfg['zoom_range'],
        width_shift_range=aug_cfg['width_shift_range'],
        height_shift_range=aug_cfg['height_shift_range'],
        shear_range=aug_cfg['shear_range'],
        horizontal_flip=aug_cfg['horizontal_flip'],
        brightness_range=aug_cfg.get('brightness_range'),
        fill_mode=aug_cfg['fill_mode'],
    )

    # No augmentation for validation/test
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=config['training']['validation_split'],
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_path,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        dataset_path,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )

    print(f"\n  📊  Train samples:      {train_gen.samples}")
    print(f"  📊  Validation samples: {val_gen.samples}")
    print(f"  📊  Classes:            {train_gen.class_indices}\n")

    return train_gen, val_gen


# ─── MODEL BUILDER ────────────────────────────────────────────────────────────

def build_model(config: dict) -> tf.keras.Model:
    """
    Build MobileNetV2 transfer learning model.

    Architecture:
        MobileNetV2 (frozen) → GAP → Dense(128) → Dropout → Dense(64) → Dense(3, softmax)
    """
    mcfg = config['model']
    input_size = tuple(config['input_size']) + (3,)
    num_classes = config['num_classes']

    # Base model — frozen
    base_model = MobileNetV2(
        input_shape=input_size,
        alpha=1.0,
        include_top=False,
        weights=mcfg['weights'],
    )
    base_model.trainable = False

    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(mcfg['dense_units_1'], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(mcfg['dropout_1'])(x)
    x = Dense(mcfg['dense_units_2'], activation='relu')(x)
    x = Dropout(mcfg['dropout_2'])(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=config['training']['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model, base_model


def unfreeze_model(model, base_model, fine_tune_at: int,
                   fine_tune_lr: float) -> tf.keras.Model:
    """Unfreeze the top layers of base model for fine-tuning."""
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers[fine_tune_at:] if l.trainable)
    print(f"  🔓  Fine-tuning {trainable} layers from layer {fine_tune_at}")

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


# ─── CALLBACKS ────────────────────────────────────────────────────────────────

def build_callbacks(config: dict, run_dir: str) -> list:
    cbk_cfg = config['callbacks']
    os.makedirs(run_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=config['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=cbk_cfg['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=cbk_cfg['reduce_lr_factor'],
            patience=cbk_cfg['reduce_lr_patience'],
            min_lr=cbk_cfg['reduce_lr_min'],
            verbose=1,
        ),
        TensorBoard(
            log_dir=run_dir,
            histogram_freq=1,
            update_freq='epoch',
        ),
        CSVLogger(
            filename=os.path.join(run_dir, 'training_log.csv'),
            append=False,
        ),
    ]
    return callbacks


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str):
    """Save training curves plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FaceMask Model Training', fontsize=16, fontweight='bold')

    metrics = [
        ('accuracy', 'val_accuracy', 'Accuracy', axes[0]),
        ('loss',     'val_loss',     'Loss',      axes[1]),
        ('precision','val_precision','Precision', axes[2]),
    ]

    for train_key, val_key, title, ax in metrics:
        if train_key in history.history:
            ax.plot(history.history[train_key], label='Train', linewidth=2)
        if val_key in history.history:
            ax.plot(history.history[val_key],   label='Val',   linewidth=2, linestyle='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊  Training plot saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_path: str):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title('Confusion Matrix — FaceMask Classifier', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊  Confusion matrix saved: {save_path}")


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def train(config: dict, fine_tune: bool = False):
    """Full training pipeline."""

    run_id  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['log_dir'], run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    print("\n" + "═"*60)
    print("  🧠  FaceMask Model Training")
    print("═"*60)
    print(f"  Run ID:      {run_id}")
    print(f"  Model:       {config['model']['base']}")
    print(f"  Epochs:      {config['training']['epochs']}")
    print(f"  Batch Size:  {config['training']['batch_size']}")
    print(f"  LR:          {config['training']['learning_rate']}")
    print(f"  Classes:     {config['classes']}")
    print(f"  Device:      {get_device_info()}")
    print("═"*60 + "\n")

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    print("📂  Loading dataset...")
    train_gen, val_gen = build_data_generators(config)

    # Model
    print("🏗️   Building model...")
    model, base_model = build_model(config)

    total_params = model.count_params()
    trainable    = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}\n")

    # Callbacks
    callbacks = build_callbacks(config, run_dir)

    # ── PHASE 1: Train head only ─────────────────────────────────
    print("🚀  Phase 1: Training classification head...")
    t0 = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1,
    )

    phase1_time = time.time() - t0
    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n  ✅  Phase 1 complete in {phase1_time/60:.1f} min")
    print(f"  📈  Best val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # ── PHASE 2: Fine-tune (optional) ────────────────────────────
    if fine_tune:
        print("\n🔧  Phase 2: Fine-tuning top layers...")
        model = unfreeze_model(
            model, base_model,
            fine_tune_at=config['training']['fine_tune_at'],
            fine_tune_lr=config['training']['fine_tune_learning_rate'],
        )
        fine_tune_callbacks = [
            cb for cb in callbacks
            if not isinstance(cb, ModelCheckpoint)
        ]
        fine_tune_callbacks.append(ModelCheckpoint(
            filepath=config['model_save_path'].replace('.h5', '_finetuned.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1,
        ))

        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config['training']['fine_tune_epochs'],
            callbacks=fine_tune_callbacks,
            verbose=1,
        )
        best_acc_ft = max(history_ft.history.get('val_accuracy', [0]))
        print(f"\n  ✅  Fine-tune complete")
        print(f"  📈  Best val accuracy: {best_acc_ft:.4f} ({best_acc_ft*100:.2f}%)")

    # ── EVALUATION ────────────────────────────────────────────────
    print("\n📊  Evaluating on validation set...")
    evaluate_model(model, val_gen, config['classes'], run_dir)

    # ── PLOTS ────────────────────────────────────────────────────
    plot_training_history(history, os.path.join(run_dir, 'training_curves.png'))

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  🎉  TRAINING COMPLETE")
    print("═"*60)
    print(f"  Model saved:  {config['model_save_path']}")
    print(f"  Logs saved:   {run_dir}")
    print(f"  Best Accuracy: {best_acc*100:.2f}%")
    print("═"*60)
    print("\nNext steps:")
    print("  1. python src/evaluate.py   — full evaluation + reports")
    print("  2. python detect_webcam.py  — run live detection")
    print("  3. python web/app.py        — launch web dashboard")
    print("  4. tensorboard --logdir logs/training  — view curves\n")

    return model, history


def evaluate_model(model, val_gen, class_names: list, save_dir: str):
    """Run evaluation and print classification report."""
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + report)

    # Save report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(save_dir, 'confusion_matrix.png'))


def get_device_info() -> str:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return f"GPU ({len(gpus)}x)"
    return "CPU"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train FaceMask Classifier')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--fine-tune', action='store_true')
    parser.add_argument('--model-save', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs:     config['training']['epochs']        = args.epochs
    if args.batch_size: config['training']['batch_size']    = args.batch_size
    if args.lr:         config['training']['learning_rate'] = args.lr
    if args.dataset:    config['dataset_path']              = args.dataset
    if args.model_save: config['model_save_path']           = args.model_save

    train(config, fine_tune=args.fine_tune)
