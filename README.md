# 📊 PRÁCTICA 3: IDENTIFICACIÓN DE FRASES CLAVE Y RESUMEN AUTOMÁTICO DE TEXTO
## Tecnologías de Lenguaje Natural

**Autor:** Escamilla Lazcano Saúl
**Grupo:** 5BV1
**Carrera:** Ingeniería En Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/Librería-spaCy-blue.svg)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/Librería-NLTK-green.svg)](https://www.nltk.org/)
[![Transformers](https://img.shields.io/badge/Librería-Transformers-yellow.svg)](https://huggingface.co/sentence-transformers)
[![Scikit-Learn](https://img.shields.io/badge/Librería-Scikit--Learn-orange.svg)](https://scikit-learn.org/stable/)

## 🚀 Descripción General del Proyecto

Este proyecto es un análisis comparativo exhaustivo de **seis algoritmos de resumen automático extractivo**, desarrollado en un Jupyter Notebook. El objetivo es procesar las cuatro primeras cartas del libro "Frankenstein", aplicar cada algoritmo para extraer las 12 oraciones más representativas y, finalmente, realizar un análisis cuantitativo y cualitativo para determinar el "mejor" algoritmo según un balance de métricas.

El proyecto demuestra cinco competencias clave:
1.  **Extracción de Texto:** Descarga y parseo del texto de "Frankenstein" desde *Project Gutenberg* usando expresiones regulares.
2.  **Normalización Justificada:** Implementación de estrategias de pre-procesamiento personalizadas para cada algoritmo, justificando por qué un enfoque único no es adecuado.
3.  **Implementación de Algoritmos:** Implementación desde cero (o con bibliotecas clave) de seis métodos de resumen: TF-IDF, Frecuencia, RAKE, TextRank, BERT y LSA.
4.  **Análisis Cuantitativo:** Medición y visualización del tiempo de ejecución, la escalabilidad y la variabilidad del rendimiento de cada método.
5.  **Análisis Cualitativo:** Evaluación de la calidad del resumen midiendo la similitud (solapamiento) entre las selecciones de oraciones y creando una rúbrica de evaluación multidimensional.

---

## 📂 1. Corpus de Datos

* **Texto de entrada:** Las cuatro primeras cartas del libro "Frankenstein" (URL de Project Gutenberg: `pg84.txt`).
* **Parámetro de resumen:** `n=12` oraciones para cada resumen.
* **Módulos de procesamiento**: NLTK, `sklearn`, `sentence-transformers` (BERT), `networkx` (TextRank), y `rake-nltk`.

## 🔡 2. Normalización de Texto (Punto 2)

Un requisito clave fue **justificar** por qué se normaliza el texto de manera diferente para cada algoritmo. Aplicar una normalización única y agresiva (como quitar toda la puntuación) es beneficioso para algunos métodos, pero perjudicial para otros.

| Algoritmo | Justificación de Normalización |
| :--- | :--- |
| **TF-IDF, Frecuencia, LSA, TextRank** | Se conserva la puntuación básica (`., !, ?`) para permitir que `sent_tokenize` de NLTK segmente las oraciones correctamente. El resto del ruido (símbolos, espacios extra) se elimina. LSA además requiere minúsculas (`.lower()`). |
| **RAKE** | Se conserva **casi toda** la puntuación. RAKE (Rapid Automatic Keyword Extraction) la utiliza como delimitador para identificar frases clave, por lo que eliminarla rompería el algoritmo. |
| **BERT** | **Normalización mínima**. Se conserva la puntuación, mayúsculas y subtítulos (`_..._`). BERT es un modelo contextual profundo que entiende el significado semántico del formato, por lo que eliminar esta información *empeoraría* sus resultados. |

---

## 🤖 3. Implementación de Algoritmos (Punto 3)

Se implementaron seis algoritmos extractivos. Todos seleccionan las 12 oraciones con mayor puntaje y las reordenan cronológicamente para mantener la coherencia.

| Algoritmo | Biblioteca/Módulo | Lógica de Puntuación (para una oración) |
| :--- | :--- | :--- |
| **TF-IDF** | `TfidfVectorizer` (sklearn) | Suma de los puntajes TF-IDF de todas las palabras que contiene. Importante si tiene palabras raras en el contexto global. |
| **Frecuencia** | `CountVectorizer` / Manual | Promedio de la frecuencia normalizada de sus palabras (excluyendo *stopwords*). Importante si contiene palabras muy comunes. |
| **RAKE** | `rake-nltk` | Suma de los puntajes RAKE de las frases clave que aparecen en ella. Importante si contiene muchas frases clave relevantes. |
| **TextRank** | `networkx` / `TfidfVectorizer` | Aplicación de PageRank sobre un grafo donde las oraciones son nodos y las aristas son su similitud (TF-IDF). Importante si es similar a otras oraciones importantes. |
| **BERT** | `SentenceTransformer` | Similitud coseno entre el vector de la oración y el vector del documento completo. Importante si su *significado semántico* es central al tema general. |
| **LSA** | `TruncatedSVD` (sklearn) | Suma de la magnitud de sus componentes (tópicos) en la matriz SVD. Importante si está fuertemente conectada a los tópicos latentes del texto. |

---

## 📈 4. Análisis y Conclusiones (Punto 4)

El análisis se dividió en dos fases para obtener una conclusión integral:

### Análisis Cuantitativo (Rendimiento)
* **Medición de Tiempos (Tabla 1, Figuras 1-4):** Se midió el tiempo de ejecución para cada carta.
* **Hallazgo Clave:** Se identificaron tres niveles de velocidad. **BERT** (0.455s prom.) es masivamente más lento que los demás. **TF-IDF** (0.003s prom.) y **LSA** (0.004s) son los más rápidos. BERT fue **140 veces más lento** que TF-IDF.
* **Escalabilidad:** El tiempo de BERT es variable y sensible a la longitud del texto, mientras que los métodos estadísticos mostraron un rendimiento casi constante.

### Análisis Cualitativo (Calidad)
* **Análisis de Características (Tabla 2):** Se verificó el éxito de la normalización (LSA fue el único en minúsculas, RAKE/BERT preservaron formato). También se demostró que `Frecuencia` tiende a seleccionar oraciones "basura" (cortas, como fechas).
* **Análisis de Similitud (Tabla 3, Figura 5):** Un mapa de calor visualizó el solapamiento (% de oraciones en común).
    * **TF-IDF y TextRank** mostraron una alta similitud (66.7%), ya que TextRank usó TF-IDF como base.
    * **BERT** demostró ser una "isla" con baja similitud (ej. 14.6% con TF-IDF), probando que su lógica de selección (semántica) es fundamentalmente única.

### Veredicto Final (Tabla 4, Figuras 6-7)
No existe un "mejor" algoritmo; la elección depende de la prioridad:

* **🏆 Mejor Calidad Semántica:** **BERT**. Es el único que "entendió" la narrativa (la aparición de la criatura en la Carta 4), pero a un costo de rendimiento extremo.
* **⚡ Mejor Velocidad y Eficiencia:** **TF-IDF**. Ideal para procesamiento masivo donde la velocidad es crítica.
* **⚖️ Mejor "Todo-Terreno" (Balance):** **RAKE**. La evaluación multidimensional (Figura 6 y 7) lo clasificó en primer lugar (21/25), mostrando un perfil equilibrado de buena velocidad, alta coherencia y excelente preservación del formato.

---

## 🚀 Cómo Ejecutar

Este proyecto es un Jupyter Notebook (`.ipynb`) y requiere un entorno compatible.

### Requisitos
* Python 3.x
* Jupyter (Lab o Notebook)
* Las bibliotecas listadas en `requirements.txt`
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

3.  **Descargar los modelos y datos necesarios:**
    ```bash
    # Descargar modelo de spaCy
    python -m spacy download en_core_web_sm
    
    # Descargar paquetes de NLTK
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    python -m nltk.downloader wordnet
    ```

4.  **Iniciar Jupyter Lab:**
    ```bash
    jupyter lab
    ```

5.  Abrir el archivo `Practica3_EscamillaLazcanoSaul_5BV1.ipynb` y ejecutar las celdas.

---

## 📄 Contenido para `requirements.txt`
(Crea un archivo `requirements.txt` y pega esto)
