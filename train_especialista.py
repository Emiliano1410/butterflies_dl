from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from etl import IMG_SIZE, load_splits, make_dataset
from especialista import PAIR, SPECIALIST_PATH
from train_tl import unfreeze_top

SRC_MODEL = "results/tl_effnetb0/best_model.keras"
OUT_DIR = Path(SPECIALIST_PATH).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pair_only(df):
    """Solo MONARCH y VICEROY, con etiqueta 0 = MONARCH, 1 = VICEROY."""
    d = df[df.label.isin(PAIR)].copy()
    d["label_idx"] = (d.label == PAIR[1]).astype(int)
    return d


def fit(model, lr, epochs, patience, train_ds, val_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True, verbose=1)])
    return pd.DataFrame(h.history)


if __name__ == "__main__":
    train_df, val_df, _, _ = load_splits()
    tr, va = pair_only(train_df), pair_only(val_df)
    print(f"Imágenes: train={len(tr)} | val={len(va)}")
    train_ds = make_dataset(tr, training=True)
    val_ds = make_dataset(va)

    # Base EfficientNetB0 del modelo general
    general = tf.keras.models.load_model(SRC_MODEL)
    base = next(l for l in general.layers if isinstance(l, tf.keras.Model))

    # Misma estructura que el modelo general pero con salida de 2 clases
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.2, value_range=(0, 1))(x)
    x = layers.Rescaling(255.0)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="especialista")

    # Fase 1: base congelada, solo se entrena la cabeza nueva
    base.trainable = False
    h1 = fit(model, 1e-3, 15, 5, train_ds, val_ds).assign(phase=1)

    # Fase 2: fine-tuning de las últimas 50 capas (BatchNorm congelado)
    unfreeze_top(base, 50)
    h2 = fit(model, 1e-5, 40, 8, train_ds, val_ds).assign(phase=2)

    model.save(SPECIALIST_PATH)
    hist = pd.concat([h1, h2], ignore_index=True)
    hist.index.name = "epoch"
    hist.to_csv(OUT_DIR / "history.csv")
    print(f"\nFase 1: {len(h1)} épocas | Fase 2: {len(h2)} épocas")
    print(f"Modelo guardado en {SPECIALIST_PATH}")
    print("Siguiente paso: python analizar_pares.py results/tl_effnetb0/best_model.keras --especialista")
