# 📊 PRÁCTICA 2: PIPELINE DE PLN Y VECTORIZACIÓN DE DOCUMENTOS
## Tecnologías de Lenguaje Natural

**Autor:** Escamilla Lazcano Saúl
**Carrera:** Ingeniería En Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/Librería-spaCy-blue.svg)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/Librería-NLTK-green.svg)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Librería-Pandas-purple.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Librería-WordCloud%20%7C%20Matplotlib-orange.svg)](https://matplotlib.org/)

## 🚀 Descripción del Proyecto

Este proyecto es un **pipeline completo de Procesamiento de Lenguaje Natural (PLN)** desarrollado en un Jupyter Notebook. El objetivo es tomar un corpus de texto crudo, aplicar un riguroso proceso de **normalización** para limpiarlo, y finalmente, **vectorizarlo** (convertirlo en números) usando cuatro técnicas fundamentales, incluyendo la implementación de **TF-IDF** desde cero.

Este cuaderno demuestra el flujo de trabajo esencial para preparar datos de texto para cualquier modelo de Machine Learning.

---

## 📂 1. Corpus de Datos

El proyecto utiliza un corpus personalizado de **10 documentos** en inglés.
* **Tema:** "Líneas de Carrera en Automovilismo" (*Racing Lines*).
* **Requisito:** Cada documento contiene más de 15 tokens para asegurar un análisis significativo.

---

## ⚙️ 2. Pipeline de Normalización de Texto (Puntos 1 y 2)

El primer paso crítico en cualquier tarea de PLN es la **normalización** del texto. Este proceso limpia el "ruido" y estandariza las palabras para que el análisis sea coherente y preciso.

Se implementaron y compararon 7 técnicas de normalización diferentes:

| Inciso | Proceso de Normalización |
| :--- | :--- |
| **a** | **Preprocesamiento Base** (Minúsculas, sin *stopwords* ni puntuación) |
| **b** | Base (a) + **Lematización Simple** (con `spaCy`) |
| **c** | Base (a) + **Stemming Simple** (con `NLTK SnowballStemmer`) |
| **d** | Base (a) → Lematización → Stemming |
| **e** | Base (a) → Stemming → Lematización |
| **f** | Base (a) + **POS-Tagging** → Lematización (con `spaCy`) |
| **g** | Base (a) + **POS-Tagging** → Stemming (con `NLTK`) |

### 🔬 Justificación de la Normalización (Punto 2)

Para los pasos siguientes, se seleccionó el **inciso (b) Lematización Simple** como el método de normalización definitivo.

**Justificación:**
* **Preservación del Significado:** A diferencia del **Stemming** (ej. `competitive` → `competit`), que simplemente "corta" las palabras, la **Lematización** las reduce a su forma base de diccionario (lema), que es una palabra real con significado (ej. `finding` → `find`). Esto es crucial para un análisis semántico preciso.
* **Evita la Sobre-reducción:** Los procesos combinados ('d' y 'e') demostraron ser redundantes, ya que el *stemming* (la operación más agresiva) anula el beneficio de la lematización.
* **Eficiencia:** El lematizador de `spaCy` (usado en 'b') ya es contextual y utiliza información de **POS-Tagging** de forma inherente, haciendo que el paso explícito ('f') sea innecesario para este caso de uso.

---

## ☁️ 3. Nube de Palabras (Punto 3)

Para validar visualmente la efectividad de nuestra normalización, se generó una **Nube de Palabras** (`WordCloud`) a partir del corpus lematizado.

Esta visualización confirma que el ruido (como "the", "a", "is") ha sido eliminado, y los términos más frecuentes son ahora los semánticamente relevantes para el tema.

**Top 10 Términos del Corpus Lematizado:**
1.  `racing` (13)
2.  `line` (12)
3.  `car` (9)
4.  `track` (8)
... y más.

*(Añade aquí el screenshot de tu nube de palabras)*
`![Nube de Palabras del Corpus 'Racing Lines'](wordcloud_racing_lines.png)`

---

## 🔢 4. Vectorización de Documentos (Punto 4)

Este es el objetivo principal: transformar los 10 documentos de texto limpio en **vectores numéricos** para que puedan ser entendidos por un algoritmo. Se implementaron 4 técnicas clave sobre un **vocabulario global de 144 términos**.

### a) One-Hot Encoding (Presencia de Término)
El método más simple. Es un vector binario (0s y 1s) donde cada índice corresponde a una palabra del vocabulario.
* **1** = la palabra **está presente** en el documento.
* **0** = la palabra **no está presente**.
* **Limitación:** Pierde toda la información de frecuencia. `car` apareciendo 1 vez o 10 veces da el mismo resultado (1).

### b) Conteo de Términos (Bolsa de Palabras / Bag of Words)
Este vector almacena el **conteo de frecuencia** de cada palabra del vocabulario en el documento.
* *Ejemplo:* Si `car` aparece 3 veces en `doc_06`, el valor en ese índice será `3`.
* **Limitación:** Da demasiado peso a palabras que son muy comunes en *todos* los documentos (como `car` en este corpus), sesgando su importancia.

### c) Probabilidad del Término (P(t))
Esta técnica crea un **único vector global** que describe la distribución de probabilidad de los términos en todo el corpus.
* **Fórmula:** $ P(t) = \frac{\text{Frecuencia de } t \text{ en todo el corpus}}{\text{Total de términos en el corpus}} $
* **Uso:** No se usa para representar documentos individuales, sino para entender la composición del corpus en su conjunto.

### d) TF-IDF (Frecuencia de Término–Frecuencia Inversa de Documento)
Es el método más robusto para ponderar la importancia de un término. Implementado desde cero, su lógica es: **"Un término es importante si es frecuente en *un* documento pero raro en *todos los demás*."**



El puntaje se calcula en dos partes:
1.  **TF (Term Frequency):** Mide la importancia local de un término en un documento.
    * $ TF(t, d) = \frac{\text{Nº de veces que } t \text{ aparece en } d}{\text{Total de términos en } d} $
2.  **IDF (Inverse Document Frequency):** Mide la rareza del término en todo el corpus.
    * $ IDF(t) = \log\left(\frac{\text{Total de documentos}}{\text{Nº de documentos que contienen } t}\right) $

El puntaje final, **TF-IDF = TF \* IDF**, penaliza palabras comunes (como `car`) dándoles un IDF bajo, y recompensa palabras específicas (como `optimal` o `bezier`) con un IDF alto. Esto proporciona una representación numérica mucho más significativa de la "firma" semántica de cada documento.

---

## 🚀 Cómo Ejecutar

Este proyecto es un Jupyter Notebook (`.ipynb`) y requiere un entorno compatible.

### Requisitos
* Python 3.x
* Jupyter (Lab o Notebook)
* Las bibliotecas listadas en `requirements.txt`.
* El modelo de lenguaje `en_core_web_sm` de `spaCy`.

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

3.  **Descargar el modelo de `spaCy` (¡Importante!):**
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
spacy
nltk
pandas
matplotlib
wordcloud
```
