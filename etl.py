from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuracion
DATA_DIR = Path("data")        
SPLITS_DIR = Path("splits")
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 42


# 1. Extracción
def extract():
    train_df = pd.read_csv(DATA_DIR / "Training_set.csv")
    test_df = pd.read_csv(DATA_DIR / "Testing_set.csv")
    return train_df, test_df


# 2. Transformacion
def check_integrity(df, img_dir):
    """Valores nulos, duplicados, archivos faltantes e imágenes corruptas."""
    print(f"{img_dir.name}: {len(df)} registros | nulos={df.isna().sum().sum()} "
          f"| duplicados={df.filename.duplicated().sum()}")
    missing, corrupt = [], []
    for name in df.filename:
        path = img_dir / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception:
            corrupt.append(name)
    print(f"  faltantes={len(missing)} | corruptas={len(corrupt)}")
    return missing, corrupt


def encode_labels(train_df):
    classes = sorted(train_df.label.unique())          # 75 clases
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_df = train_df.assign(label_idx=train_df.label.map(class_to_idx))
    return train_df, classes


def split_stratified(df):
    """70% train / 15% val / 15% test, conservando la proporcion de clases."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df.label_idx, random_state=SEED)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp.label_idx, random_state=SEED)
    return train, val, test


# 3. Carga
def save_splits(train, val, test, classes):
    SPLITS_DIR.mkdir(exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)
    pd.Series(classes).to_csv(SPLITS_DIR / "classes.csv", index=False, header=False)


def load_splits():
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    classes = pd.read_csv(SPLITS_DIR / "classes.csv", header=None)[0].tolist()
    return train, val, test, classes


def make_dataset(df, img_dir=DATA_DIR / "train", training=False, labeled=True):
    """tf.data pipeline: lee JPG -> resize 256x256 -> [0,1] -> batches."""
    import tensorflow as tf

    # siempre se indica la carpeta porque los nombres Image_N.jpg se repiten entre train y test
    paths = [str(img_dir / f) for f in df.filename]

    def load(path, label=None):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, IMG_SIZE) / 255.0
        return (img, label) if labeled else img

    if labeled:
        ds = tf.data.Dataset.from_tensor_slices((paths, df.label_idx.values))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    train_df, test_df = extract()
    check_integrity(train_df, DATA_DIR / "train")
    check_integrity(test_df, DATA_DIR / "test")

    train_df, classes = encode_labels(train_df)
    print(f"clases: {len(classes)}")

    train, val, test = split_stratified(train_df)
    print(f"train={len(train)} | val={len(val)} | test={len(test)}")
    save_splits(train, val, test, classes)
