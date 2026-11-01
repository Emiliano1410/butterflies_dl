from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from etl import load_splits, make_dataset, IMG_SIZE

# Configuracion
RUN_NAME = "tl_effnetb0"
EPOCHS_FROZEN = 20          # fase 1
EPOCHS_FINETUNE = 40        # fase 2 (early stopping)
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_LAYERS = 50       # ultimas capas de la base que se descongelan en la fase 2
OUT_DIR = Path("results") / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPU:", tf.config.list_physical_devices("GPU"))


# Modelo
def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    # Mismo aumento de datos que cnn_v2 (las imagenes llegan en [0, 1])
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.2, value_range=(0, 1))(x)
    # EfficientNet trae su propia normalizacion y espera pixeles en [0, 255]
    x = layers.Rescaling(255.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=RUN_NAME), base


def unfreeze_top(base, n):
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:               # BatchNorm se mantiene congelado
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def callbacks(best_val_acc=None):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            OUT_DIR / "best_model.keras", monitor="val_accuracy", save_best_only=True,
            initial_value_threshold=best_val_acc),   # fase 2 solo guarda si supera a la fase 1
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True, verbose=1),
    ]


# Graficas
def plot_history(hist, phase_change):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        ax.plot(hist[metric], label="Train")
        ax.plot(hist[f"val_{metric}"], label="Validation")
        ax.axvline(phase_change - 0.5, color="gray", linestyle="--", label="Inicio fine-tuning")
        ax.set_title(f"{title} - {RUN_NAME}")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curvas.png", dpi=150)


if __name__ == "__main__":
    train_df, val_df, test_df, classes = load_splits()
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    model, base = build_model(len(classes))

    # Fase 1: extraccion de caracteristicas
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
                   callbacks=callbacks())
    h1 = pd.DataFrame(h1.history).assign(phase=1)

    # Fase 2: fine-tuning
    unfreeze_top(base, FINE_TUNE_LAYERS)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(f"\nFine-tuning: parámetros entrenables = "
          f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
                   callbacks=callbacks(h1.val_accuracy.max()))
    h2 = pd.DataFrame(h2.history).assign(phase=2)

    hist = pd.concat([h1, h2], ignore_index=True)
    hist.index.name = "epoch"
    hist.to_csv(OUT_DIR / "history.csv")
    plot_history(hist, len(h1))
    print(f"\nFase 1: {len(h1)} épocas | Fase 2: {len(h2)} épocas")

    # Evaluacion final con el mejor modelo de ambas fases
    best = tf.keras.models.load_model(OUT_DIR / "best_model.keras")
    print("\nResultados (mejor época):")
    for name, ds in (("train", make_dataset(train_df)), ("val", val_ds), ("test", test_ds)):
        loss, acc = best.evaluate(ds, verbose=0)
        print(f"  {name:5s}  loss={loss:.4f}  accuracy={acc:.4f}")
