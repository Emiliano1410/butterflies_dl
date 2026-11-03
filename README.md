# Clasificación de especies de mariposas con CNN

Clasificador de 75 especies de mariposas con TensorFlow/Keras. Se compara una CNN entrenada desde cero contra transfer learning con EfficientNetB0 (ImageNet), y el modelo final se usa en una interfaz gráfica con Gradio.

Dataset: [Butterfly Image Classification (Kaggle)](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification): 9,285 imágenes, 75 clases.

## Estructura

```
etl.py           Verificación de datos, división estratificada 70/15/15 y pipeline tf.data
train_cnn.py     CNN desde cero (IMPROVED = False: línea base | True: versión mejorada)
train_tl.py      Transfer learning con EfficientNetB0 (fase congelada + fine-tuning)
app.py           Interfaz gráfica de predicción (Gradio)
splits/          Archivos de cada división y lista de clases
results/         Modelos entrenados, historiales y curvas de cada corrida
data/            Dataset (no incluido en el repo, descargar de Kaggle)
```

## Dataset

Descarga el dataset de Kaggle y descomprímelo en `data/`, de forma que queden:

```
data/train/   data/test/   data/Training_set.csv   data/Testing_set.csv
```

## Ejecutar la interfaz (cualquier computadora, sin GPU)

Requiere Python 3.10 a 3.13. La predicción funciona en CPU.

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements-demo.txt
python app.py
```

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-demo.txt
python app.py
```

Abrir http://127.0.0.1:7860 en el navegador.

La pestaña **Subir imagen** solo necesita el modelo. Las pestañas **Muestra del set de test** y **Set de test completo** necesitan además la carpeta `data/` con el dataset.

## Entrenar los modelos (Linux o WSL2 con GPU NVIDIA)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python etl.py          # genera splits/
python train_cnn.py    # CNN desde cero
python train_tl.py     # EfficientNetB0
```

Entrenado con una RTX 4070 Ti, TensorFlow 2.21 y Python 3.12 en WSL2.

## Autor

Emiliano Camacho Ponce, Tecnológico de Monterrey
