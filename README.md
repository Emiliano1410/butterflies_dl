# Clasificación de especies de mariposas con CNN

Clasificador de 75 especies de mariposas con TensorFlow/Keras. Se compara una CNN entrenada desde cero contra transfer learning con EfficientNetB0 (ImageNet), y el modelo final se usa en una interfaz gráfica.

Dataset: [Butterfly Image Classification (Kaggle)](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification): 9,285 imágenes, 75 clases.

## Estructura

```
etl.py           Verificación de datos, división estratificada 70/15/15 y pipeline tf.data
train_cnn.py     CNN desde cero
train_tl.py      Transfer learning con EfficientNetB0
app.py           Interfaz gráfica de predicción
splits/          Archivos de cada división y lista de clases
results/         Modelos entrenados, historiales y curvas de cada corrida
data/            Dataset (no incluido en el repo, descargar de Kaggle)
```

## Autor

Emiliano Camacho Ponce, Tecnológico de Monterrey
