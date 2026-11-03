import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf

from etl import DATA_DIR, IMG_SIZE, load_splits, make_dataset

MODEL_PATH = "results/tl_effnetb0/best_model.keras"
TEST_DIR = DATA_DIR / "test"
N_SAMPLE = 8

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model(MODEL_PATH)
_, _, _, CLASSES = load_splits()
test_files = pd.read_csv(DATA_DIR / "Testing_set.csv")


def preprocess(img):
    """Mismo preprocesamiento que el ETL: resize 256x256 y escala a [0, 1]."""
    return tf.image.resize(img, IMG_SIZE) / 255.0


def predict_image(img):
    """Imagen subida -> top 5 especies con su probabilidad."""
    if img is None:
        return None
    probs = model.predict(preprocess(img)[None, ...], verbose=0)[0]
    top5 = np.argsort(probs)[::-1][:5]
    return {CLASSES[i]: float(probs[i]) for i in top5}


def predict_sample():
    """N imágenes al azar de la carpeta test (sin etiqueta) con su predicción."""
    sample = test_files.sample(N_SAMPLE)
    paths = [str(TEST_DIR / f) for f in sample.filename]
    imgs = [preprocess(tf.io.decode_jpeg(tf.io.read_file(p), channels=3)) for p in paths]
    probs = model.predict(tf.stack(imgs), verbose=0)
    return [(p, f"{CLASSES[pr.argmax()]} ({pr.max():.0%})") for p, pr in zip(paths, probs)]


def predict_test_set():
    """Predice las 2,786 imágenes de test en el orden de Testing_set.csv y guarda un CSV."""
    ds = make_dataset(test_files, img_dir=TEST_DIR, labeled=False)
    probs = model.predict(ds, verbose=0)
    out = test_files.assign(label=[CLASSES[i] for i in probs.argmax(axis=1)],
                            confidence=probs.max(axis=1).round(4))
    out.to_csv("predicciones_test.csv", index=False)
    return out.head(20), "predicciones_test.csv"


with gr.Blocks(title="Clasificador de mariposas") as demo:
    gr.Markdown("# Clasificador de especies de mariposas\n"
                "EfficientNetB0 con transfer learning · 75 especies · 91.7% de accuracy en test")

    with gr.Tab("Subir imagen"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Imagen de mariposa")
            label_out = gr.Label(num_top_classes=5, label="Top 5 especies")
        img_in.change(predict_image, img_in, label_out)
        gr.Examples([str(TEST_DIR / f) for f in test_files.filename[:6]], inputs=img_in)

    with gr.Tab("Muestra del set de test"):
        btn_sample = gr.Button(f"Predecir {N_SAMPLE} imágenes al azar")
        gallery = gr.Gallery(label="Especie predicha (confianza)", columns=4)
        btn_sample.click(predict_sample, None, gallery)

    with gr.Tab("Set de test completo"):
        btn_all = gr.Button("Predecir las 2,786 imágenes y guardar CSV")
        table = gr.Dataframe(label="Primeras 20 predicciones")
        csv_out = gr.File(label="predicciones_test.csv")
        btn_all.click(predict_test_set, None, [table, csv_out])

if __name__ == "__main__":
    demo.launch()
