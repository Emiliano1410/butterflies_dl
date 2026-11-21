import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from etl import DATA_DIR, load_splits, make_dataset
from especialista import PAIR, SPECIALIST_PATH, refinar

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL_PATH = Path(args[0] if args else "results/tl_effnetb0/best_model.keras")
USE_SPECIALIST = "--especialista" in sys.argv
OUT = Path(SPECIALIST_PATH).parent if USE_SPECIALIST else MODEL_PATH.parent

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

_, val_df, test_df, classes = load_splits()
model = tf.keras.models.load_model(MODEL_PATH)
specialist = tf.keras.models.load_model(SPECIALIST_PATH) if USE_SPECIALIST else None
idx = {c: classes.index(c) for c in PAIR}
other = {PAIR[0]: PAIR[1], PAIR[1]: PAIR[0]}

rows, errors = [], []
for split, df in (("val", val_df), ("test", test_df)):
    probs = model.predict(make_dataset(df), verbose=0)
    if USE_SPECIALIST:
        probs = refinar(probs, specialist.predict(make_dataset(df), verbose=0), classes)
    pred, true = probs.argmax(axis=1), df.label_idx.values
    print(f"\n[{split}] accuracy global: {(pred == true).mean():.4f}")
    for c in PAIR:
        mask = true == idx[c]
        rows.append({
            "split": split, "clase": c, "imagenes": int(mask.sum()),
            "correctas": int((pred[mask] == idx[c]).sum()),
            f"confundida_con_par": int((pred[mask] == idx[other[c]]).sum()),
            "confundida_con_otra": int(((pred[mask] != idx[c]) & (pred[mask] != idx[other[c]])).sum()),
            "falsos_positivos": int(((pred == idx[c]) & ~mask).sum()),   # otras especies predichas como c
        })
        r = rows[-1]
        print(f"  {c:8s}: {r['correctas']}/{r['imagenes']} correctas | "
              f"como {other[c]}: {r['confundida_con_par']} | como otra: {r['confundida_con_otra']} | "
              f"falsos positivos: {r['falsos_positivos']}")
    # imágenes de la pareja mal clasificadas
    for i in np.where(np.isin(true, list(idx.values())) & (pred != true))[0]:
        errors.append((split, df.filename.iloc[i], classes[true[i]], classes[pred[i]], probs[i].max()))
    if split == "test":   # los 5 pares de especies más confundidos en todo el test
        cm = tf.math.confusion_matrix(true, pred, num_classes=len(classes)).numpy()
        np.fill_diagonal(cm, 0)
        top = np.dstack(np.unravel_index(np.argsort(cm, axis=None)[::-1][:5], cm.shape))[0]
        print("\n  Pares más confundidos en test (real -> predicha):")
        for t, p in top:
            if cm[t, p]:
                print(f"    {classes[t]} -> {classes[p]}: {cm[t, p]}")

report = pd.DataFrame(rows)
report.to_csv(OUT / "pair_report.csv", index=False)
print(f"\nReporte guardado en {OUT / 'pair_report.csv'}")

if errors:
    errors = errors[:12]                       # máximo 12 imágenes, en filas de 6
    cols = min(len(errors), 6)
    rows_n = -(-len(errors) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3 * cols, 4 * rows_n), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (split, f, t, p, conf) in zip(axes.flat, errors):
        ax.imshow(plt.imread(DATA_DIR / "train" / f))
        ax.set_title(f"Real: {t}\nPred: {p} ({conf:.0%})\n[{split}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "errores_par.png", dpi=150, bbox_inches="tight")   # sin recortar títulos
    print(f"Figura de errores guardada en {OUT / 'errores_par.png'}")
else:
    print("Sin errores en las imágenes de MONARCH y VICEROY.")