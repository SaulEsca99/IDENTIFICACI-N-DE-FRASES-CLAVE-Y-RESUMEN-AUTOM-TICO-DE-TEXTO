# 📊 PRÁCTICA 3: IMPLEMENTACIÓN Y EVALUACIÓN DE NAÏVE BAYES
## Tecnologías de Lenguaje Natural

**Autor:** Escamilla Lazcano Saúl
**Grupo:** 5BV1
**Carrera:** Ingeniería En Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Librería-Pandas-purple.svg)](https://pandas.pydata.org/)
[![spaCy](https://img.shields.io/badge/Librería-spaCy-blue.svg)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/Librería-NLTK-green.svg)](https://www.nltk.org/)
[![Scikit-learn](https://img.shields.io/badge/Librería-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Librería-Seaborn%20%7C%20Matplotlib-blueviolet.svg)](https://seaborn.pydata.org/)

## 🚀 Descripción del Proyecto

Este proyecto es una implementación completa del algoritmo clasificador **Bayesiano Ingenuo (Naïve Bayes)** **desde cero** en Python. El objetivo es construir un modelo de **Análisis de Sentimiento** capaz de predecir si una reseña de película es "positiva" o "negativa" basándose únicamente en su contenido textual.

El *pipeline* del proyecto cubre todos los pasos esenciales de una tarea de PLN:
1.  **Carga y Exploración de Datos** del dataset IMDB.
2.  **Preprocesamiento y Normalización de Texto** avanzado usando `spaCy` y `NLTK`.
3.  **Implementación del Modelo** (`NaiveBayesPersonalizado`) desde cero.
4.  **Entrenamiento y Evaluación** del modelo con métricas de clasificación estándar.
5.  **Visualización de Resultados**, incluyendo una matriz de confusión y nubes de palabras.

## 💾 1. Dataset

El conjunto de datos utilizado es el **"IMDB Dataset of 50K Movie Reviews"**. Este es un corpus canónico para tareas de clasificación binaria de sentimiento.
* **Archivo:** `IMDB Dataset.csv`
* **Tamaño:** 50,000 reseñas.
* **Clases:** "positiva" (25,000) y "negativa" (25,000).

## ⚙️ 2. Pipeline de Preprocesamiento de Texto

Antes de entrenar, el texto crudo debe ser normalizado. Se implementaron dos métodos de normalización para comparar: *Stemming* (con `NLTK`) y *Lematización* (con `spaCy`).

Se seleccionó la **Lematización** para el pipeline final, ya que produce palabras léxicamente correctas (lemas), lo que es más preciso que las raíces generadas por el *stemming*.

El pipeline de normalización (`lematizar_texto`) incluye:
1.  **Conversión a Minúsculas:** `texto.lower()`
2.  **Eliminación de HTML:** Se usó `re` para eliminar etiquetas (ej. `<br />`).
3.  **Tokenización (spaCy):** Se procesa el texto con el modelo `en_core_web_sm`.
4.  **Eliminación de Stopwords y Puntuación:** Se filtran palabras comunes y signos de puntuación.
5.  **Lematización (spaCy):** Cada token se reduce a su forma base de diccionario (ej. "running" → "run").

## 🧠 3. Implementación: Naïve Bayes desde Cero

El núcleo de la práctica es la clase `NaiveBayesPersonalizado`, que no utiliza las implementaciones de `sklearn` para el clasificador.

### A. Entrenamiento (`fit`)
El método `fit` aprende las probabilidades necesarias del corpus de entrenamiento (`X_train`, `y_train`).

**1. Cálculo de Priors de Clase $P(c)$:**
Calcula la probabilidad base de cada clase (positiva o negativa) en el dataset.
$$ P(c) = \frac{\text{Documentos en la clase } c}{\text{Total de documentos}} $$

**2. Cálculo de Probabilidades Condicionales (Likelihoods) $P(w|c)$:**
Calcula la probabilidad de que una palabra $w$ aparezca, dado que pertenece a una clase $c$.

* **Conteo de Palabras:** Se construye un vocabulario de frecuencia para cada clase.
* **Suavizado de Laplace (Add-1):** Se aplica un suavizado (con $\alpha = 1$) para manejar palabras que aparecen en el set de prueba pero no en el de entrenamiento. Esto evita probabilidades de cero que anularían todo el cálculo.

La fórmula de probabilidad de una palabra con suavizado es:
$$ P(w_i | c) = \frac{\text{frecuencia}(w_i, c) + \alpha}{\text{Total de palabras en } c + \alpha \cdot |V|} $$
Donde $|V|$ es el tamaño del vocabulario global.

### B. Predicción (`predict`)
El método `predict` clasifica nuevos documentos. Para evitar el **underflow numérico** (multiplicar muchas probabilidades pequeñas da como resultado cero), se utiliza la **suma de log-probabilidades**. El teorema de Bayes en su forma logarítmica es:

$$ c_{\text{pred}} = \underset{c}{\operatorname{argmax}} \left( \log(P(c)) + \sum_{i=1}^{n} \log(P(w_i | c)) \right) $$

El modelo asigna la clase $c$ (positiva o negativa) que maximice esta suma.



## 📊 4. Evaluación y Resultados

El modelo se entrenó con el 80% de los datos y se evaluó con el 20% restante.

### Métricas de Desempeño
La evaluación (`sklearn.metrics`) arrojó un rendimiento excelente:

* **Accuracy (Exactitud):** **~86.2%**
* **Reporte de Clasificación:**
    | Clase | Precision | Recall | F1-Score |
    | :--- | :--- | :--- | :--- |
    | Negativa | 0.86 | 0.87 | 0.86 |
    | Positiva | 0.87 | 0.86 | 0.86 |

### Matriz de Confusión
La matriz de confusión (visualizada con `seaborn`) confirma el buen desempeño del modelo, mostrando una alta concentración de predicciones correctas en la diagonal principal.

*(**Instrucción:** Sube tu imagen de la matriz de confusión al repositorio y nómbrala `matriz_confusion.png` para que aparezca aquí)*
`![Matriz de Confusión](matriz_confusion.png)`

### Nubes de Palabras
Se generaron nubes de palabras (`wordcloud`) a partir de los vocabularios aprendidos por el modelo para cada clase, mostrando visualmente los términos más distintivos de cada sentimiento.

*(**Instrucción:** Sube tus nubes de palabras y nómbralas como se sugiere)*
| Nube de Palabras Positivas | Nube de Palabras Negativas |
| :---: | :---: |
| `![Nube de Palabras Positivas](wordcloud_positiva.png)` | `![Nube de Palabras Negativas](wordcloud_negativa.png)` |

## 💡 Conclusión

Esta práctica demostró con éxito la implementación de un clasificador Naïve Bayes Multinomial desde cero. Los resultados de exactitud (~86%) son muy buenos y demuestran la efectividad de este algoritmo para tareas de clasificación de texto. El uso de un pipeline de normalización robusto (especialmente la lematización) y técnicas como el suavizado de Laplace fueron cruciales para el rendimiento del modelo.

---

## 🚀 Cómo Ejecutar

Este proyecto es un Jupyter Notebook (`.ipynb`) y requiere un entorno compatible.

### Requisitos
* Python 3.x
* Jupyter (Lab o Notebook)
* Las bibliotecas listadas en `requirements.txt`.
* El modelo de lenguaje `en_core_web_sm` de `spaCy`.
* **Importante:** El archivo del dataset `IMDB Dataset.csv` debe estar en la misma carpeta.

### Pasos de Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd [NOMBRE-DEL-REPOSITORIO]
    ```

2.  **Instalar dependencias:**
    (Se recomienda crear un entorno virtual: `python -m venv venv` y `source venv/bin/activate`)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descargar el modelo de `spaCy`:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Iniciar Jupyter Lab:**
    ```bash
    jupyter lab
    ```

5.  Abrir el archivo `Practica3_EscamillaLazcanoSaul_5BV1.ipynb` y ejecutar las celdas.

### 🐛 Solución de Errores Comunes

**Error (el de tu imagen):** `ModuleNotFoundError: No module named 'wordcloud'`

**Solución:** Este error significa que la biblioteca `wordcloud` no está instalada en tu entorno. Para arreglarlo:

1.  Abre tu terminal.
2.  Activa tu entorno de Conda (ej. `conda activate nlp_env`).
3.  Ejecuta el siguiente comando:
    ```bash
    conda install -c conda-forge wordcloud
    ```
4.  **Reinicia el kernel** de tu Jupyter Notebook y vuelve a ejecutar las celdas.

---

## 📄 Contenido para `requirements.txt`
(Crea un archivo `requirements.txt` y pega esto)
```
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
spacy
scikit-learn
```
