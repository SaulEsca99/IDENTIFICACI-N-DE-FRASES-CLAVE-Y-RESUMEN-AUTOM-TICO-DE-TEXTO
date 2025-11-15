# 📊 PRÁCTICA 3: IMPLEMENTACIÓN Y EVALUACIÓN DE NAÏVE BAYES
## Tecnologías de Lenguaje Natural

**Autor:** Escamilla Lazcano Saúl
**Carrera:** Ingeniería En Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/Librería-spaCy-blue.svg)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/Librería-NLTK-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Librería-Pandas-purple.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Librería-Scikit--learn-orange.svg)](https://scikit-learn.org/)

## 🚀 Descripción del Proyecto

Este proyecto es una implementación completa del algoritmo **Clasificador Bayesiano Ingenuo (Naïve Bayes)** desde cero en Python, aplicado a un problema de **Análisis de Sentimiento**. El objetivo es predecir si una reseña de película es "positiva" o "negativa" basándose en su contenido textual.

El proyecto abarca todo el pipeline de un proyecto de PLN:
1.  **Carga y Exploración de Datos** del dataset IMDB.
2.  **Preprocesamiento y Normalización de Texto** avanzado usando `spaCy` y `NLTK`.
3.  **Implementación del Modelo** `NaiveBayesPersonalizado` desde cero.
4.  **Entrenamiento y Evaluación** del modelo usando métricas estándar de clasificación.
5.  **Visualización de Resultados**, incluyendo una matriz de confusión y nubes de palabras.

## 💾 1. Dataset

El conjunto de datos utilizado es el **"IMDB Dataset of 50K Movie Reviews"**. Este es un corpus estándar para tareas de clasificación binaria de sentimiento.
* **Tamaño:** 50,000 reseñas.
* **Clases:** "positiva" (25,000) y "negativa" (25,000).
* **Objetivo:** Clasificar el sentimiento de la reseña.

## ⚙️ 2. Pipeline de Preprocesamiento de Texto

Antes de entrenar, el texto crudo debe ser normalizado. Se implementaron dos métodos de normalización para comparar: *Stemming* (con `NLTK`) y *Lematización* (con `spaCy`).

Se seleccionó la **Lematización** para el pipeline final, ya que produce palabras léxicamente correctas (lemas), lo que es más preciso que las raíces generadas por el *stemming*.

El pipeline de normalización (`lematizar_texto`) incluye:
1.  **Conversión a Minúsculas:** `texto.lower()`
2.  **Eliminación de HTML:** Se usó `re` para eliminar etiquetas HTML (ej. `<br />`).
3.  **Tokenización (spaCy):** Se procesa el texto con el modelo `en_core_web_sm`.
4.  **Eliminación de Stopwords y Puntuación:** Se filtran palabras comunes y signos de puntuación.
5.  **Lematización (spaCy):** Cada token se reduce a su forma base (ej. "running" → "run").

## 🧠 3. ImplementACIÓN: Naïve Bayes desde Cero

El núcleo de la práctica es la clase `NaiveBayesPersonalizado`, que no utiliza las implementaciones de `sklearn` para el clasificador.

### A. Entrenamiento (`fit`)

El método `fit` aprende las probabilidades necesarias del corpus de entrenamiento.

**1. Cálculo de Priors de Clase $P(c)$:**
Se calcula la probabilidad de que un documento pertenezca a una clase (positiva o negativa) sin ver el texto.
$$ P(c) = \frac{\text{Documentos en la clase } c}{\text{Total de documentos}} $$

**2. Cálculo de Probabilidades de Palabras (Likelihoods) $P(w|c)$:**
Se calcula la probabilidad de que una palabra $w$ aparezca, dado que pertenece a una clase $c$.

* **Conteo de Palabras:** Se construye un vocabulario de frecuencia para cada clase.
* **Suavizado de Laplace (Add-1):** Para manejar palabras que no se vieron en el entrenamiento (y evitar probabilidades de cero), se aplica el suavizado de Laplace ($ \alpha = 1 $).

La fórmula para la probabilidad de una palabra con suavizado es:
$$ P(w_i | c) = \frac{\text{frecuencia}(w_i, c) + \alpha}{\text{Total de palabras en } c + \alpha \cdot |V|} $$
Donde $|V|$ es el tamaño del vocabulario global.

### B. Predicción (`predict`)

El método `predict` clasifica nuevos documentos. Para evitar el *underflow* numérico (multiplicar muchas probabilidades pequeñas), se utiliza la suma de **log-probabilidades**:

$$ c_{\text{pred}} = \underset{c}{\operatorname{argmax}} \left( \log(P(c)) + \sum_{i=1}^{n} \log(P(w_i | c)) \right) $$

El documento se asigna a la clase $c$ que maximice esta suma.

## 📊 4. Evaluación y Resultados

El modelo se entrenó con el 80% del dataset y se evaluó con el 20% restante.

### Métricas de Desempeño
La evaluación (`sklearn.metrics`) arrojó excelentes resultados:

* **Accuracy (Exactitud):** ~86.2%
* **Reporte de Clasificación:**
    | Clase | Precision | Recall | F1-Score |
    | :--- | :--- | :--- | :--- |
    | Negativa | 0.86 | 0.87 | 0.86 |
    | Positiva | 0.87 | 0.86 | 0.86 |

### Matriz de Confusión
La matriz de confusión (visualizada con `seaborn`) muestra cómo se distribuyeron las predicciones correctas e incorrectas.



### Nubes de Palabras
Se generaron nubes de palabras (`wordcloud`) a partir del vocabulario aprendido por el modelo para las clases "positiva" y "negativa", mostrando los términos más distintivos de cada sentimiento.

| Nube Positiva | Nube Negativa |
| :---: | :---: |
|  |  |

## 💡 Conclusión

Esta práctica demostró con éxito la implementación de un clasificador Naïve Bayes Multinomial desde cero. Los resultados de exactitud (~86%) son muy buenos y demuestran la efectividad de este algoritmo para tareas de clasificación de texto. El uso de un pipeline de normalización robusto (especialmente la lematización) y técnicas como el suavizado de Laplace fueron cruciales para el rendimiento del modelo.

---

## 🚀 Cómo Ejecutar

Este proyecto es un Jupyter Notebook (`.ipynb`) y requiere un entorno compatible.

### Requisitos
* Python 3.x
* Jupyter (Lab o Notebook)
* Las bibliotecas listadas en `requirements.txt`
* El modelo de lenguaje `en_core_web_sm` de `spaCy`.
* El dataset `IMDB Dataset.csv` (no incluido en este repo, debe ser descargado).

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

5.  Abrir el archivo `.ipynb` y ejecutar las celdas.

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