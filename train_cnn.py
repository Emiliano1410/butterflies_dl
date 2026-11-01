from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from etl import load_splits, make_dataset, IMG_SIZE

# Configuracion
IMPROVED = True          # False = base | True = aumento de datos + dropout + LR adaptativo
RUN_NAME = "cnn_v2" if IMPROVED else "cnn_base"
EPOCHS = 100 if IMPROVED else 30 
LEARNING_RATE = 1e-3
OUT_DIR = Path("results") / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reservar memoria de la GPU
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPU:", tf.config.list_physical_devices("GPU"))


# Modelo
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


def build_cnn(num_classes):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = inputs
    if IMPROVED:
        # Aumento de datos (solo durante el entrenamiento)
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)                 # ±36°
        x = layers.RandomZoom(0.15)(x)
        x = layers.RandomContrast(0.2, value_range=(0, 1))(x)
    for filters in (32, 64, 128, 256):      # 256 -> 128 -> 64 -> 32 -> 16 px
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    if IMPROVED:
        # Dropout
        x = layers.Dropout(0.3)(x)                        
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=RUN_NAME)


# Graficas
def plot_history(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        ax.plot(hist[metric], label="Train")
        ax.plot(hist[f"val_{metric}"], label="Validation")
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

    model = build_cnn(len(classes))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Se guarda el modelo del mejor epoch segun accuracy de validation
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        OUT_DIR / "best_model.keras", monitor="val_accuracy", save_best_only=True)
    callbacks = [checkpoint]
    if IMPROVED:
        # Reducir LR a la mitad si val_loss no mejora en 5 epochs + early stopping
        callbacks += [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=5, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                             restore_best_weights=True, verbose=1),
        ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=callbacks)

    hist = pd.DataFrame(history.history)
    hist.index.name = "epoch"
    hist.to_csv(OUT_DIR / "history.csv")
    plot_history(hist)

    # Evaluacion final con el mejor modelo
    best = tf.keras.models.load_model(OUT_DIR / "best_model.keras")
    print("\nResultados (mejor época):")
    for name, ds in (("train", make_dataset(train_df)), ("val", val_ds), ("test", test_ds)):
        loss, acc = best.evaluate(ds, verbose=0)
        print(f"  {name:5s}  loss={loss:.4f}  accuracy={acc:.4f}")