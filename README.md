# Taller de Clasificación con Árboles de Decisión
**Curso de Inteligencia Artificial**  
*De las preguntas de un árbol a la evaluación de sus predicciones*

Solución integral, reproducible y con arquitectura de software modular orientada a objetos (principios SOLID) para el taller de árboles de decisión.

---

## 🏛️ Arquitectura Modular del Software

El proyecto se estructuró bajo estándares de ingeniería de software profesional, desacoplando datos, matemáticas de impureza, modelos manuales, validación algorítmica, experimentación y generación de reportes:

```text
Taller Arboles de decisión/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constantes, semillas, hiperparámetros y rutas
│   ├── synthetic/                # Módulo para el Caso Sintético de Diamantes
│   │   ├── __init__.py
│   │   ├── data.py               # Definición formal de datos (12 diamantes y N1-N4)
│   │   ├── gini.py               # Fórmulas de Gini, partición ponderada y evaluación de splits
│   │   ├── manual_tree.py        # Modelo OOP del árbol manual, reglas SI... ENTONCES... y trazador
│   │   ├── metrics.py            # Métricas binarias (VP, VN, FP, FN), costos y desbalance 95/5
│   │   └── sklearn_verifier.py   # Verificación con DecisionTreeClassifier y comparación
│   ├── iris/                     # Módulo para el Experimento con Dataset Iris
│   │   ├── __init__.py
│   │   ├── dataset.py            # Carga y partición estratificada 60% Train / 20% Val / 20% Test
│   │   ├── experiment.py         # Entrenamiento de candidatos (1, 3, None) y regla de selección
│   │   ├── evaluation.py         # Evaluación en Test y benchmark contra DummyClassifier
│   │   └── visualization.py      # Diagramas del árbol seleccionado y matriz de confusión
│   └── report/                   # Módulo de Entregables
│       ├── __init__.py
│       ├── pdf_generator.py      # Generador de informe formal de 8 páginas en ReportLab
│       └── notebook_generator.py # Generador programático del Jupyter Notebook (.ipynb)
├── figures/                      # Gráficos generados en alta resolución (300 DPI)
│   ├── manual_tree_diagram.png   # Diagrama arquitectónico del árbol manual
│   ├── sklearn_diamond_tree.png  # Árbol de Scikit-Learn sobre diamantes
│   ├── iris_best_tree.png        # Árbol ganador sobre Iris (max_depth=None, Efectiva=4)
│   └── iris_confusion_matrix.png # Matriz de confusión en conjunto de prueba
├── main.py                       # Orquestador y punto de entrada CLI
├── taller_arboles.ipynb          # Notebook interactivo y ejecutable
├── Informe_Taller_Arboles.pdf    # Informe formal completo para entrega
├── Taller_Arboles.pdf            # Enunciado original del taller
└── README.md                     # Documentación técnica del proyecto
```

---

## 🚀 Instrucciones de Ejecución

### 1. Ejecución Automatizada de la Pipeline Completa
Para ejecutar todos los cálculos, generar todas las gráficas, crear el Jupyter Notebook y compilar el informe PDF formal:

```bash
python main.py
```

### 2. Uso Interactivo en Jupyter Notebook
Puede abrir y ejecutar `taller_arboles.ipynb` directamente en Visual Studio Code, Jupyter Lab o Google Colab:
- Los datos sintéticos están embebidos en el código.
- Iris se carga directamente desde `sklearn.datasets.load_iris()`.
- No requiere descargar archivos externos de internet.

---

## 📊 Resumen de Resultados Técnicos

### 1. Formulación del Problema (Sección 1)
- **Unidad de observación:** Un diamante individual.
- **Predictores $X$:** $x_1$ = Quilates (peso continuo), $x_2$ = Profundidad (%) (porcentaje continuo).
- **Etiqueta $y$:** $y \in \{A, E\}$ donde $A$ = Aceptable (clase positiva conmutada) y $E$ = Excelente.
- **Fuga de datos (Leakage):** El campo *"resultado final de la inspección"* **NO** debe usarse porque se genera después/durante la clasificación, violando la disponibilidad previa en inferencia. El precio solo puede usarse si está fijado antes de emitir la clase y no depende de ella.

### 2. Selección de la Raíz con Gini (Sección 2)
Impureza de la raíz: $G(S) = 1 - (0.5^2 + 0.5^2) = \mathbf{0.5000}$.

| Pregunta Candidata | $n_L$ | $A/E$ en $S_L$ | $G(S_L)$ | $n_R$ | $A/E$ en $S_R$ | $G(S_R)$ | $G_{div}$ | $\Delta G$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Quilates <= 0.75` | 5 | 4 / 1 | 0.3200 | 7 | 2 / 5 | 0.4082 | 0.3714 | 0.1286 |
| `Profundidad <= 63.5` | 8 | 2 / 6 | 0.3750 | 4 | 4 / 0 | 0.0000 | 0.2500 | **0.2500** |

- **Decisión:** Se elige **`Profundidad <= 63.5`** por maximizar la ganancia ($\Delta G = 0.2500$) y aislar una rama derecha 100% pura ($G=0.0000$, 4 diamantes Aceptables).
- **Ponderación:** No se pueden promediar impurezas sin considerar tamaños porque cada partición tiene diferente proporción de datos ($|S_L| \neq |S_R|$); un promedio simple daría igual peso a hojas insignificantes.

### 3. Construcción del Árbol e Inferencia (Sección 3)
- **Hijo impuro izquierdo ($S_L$):** Al evaluar puntos medios entre los valores de Quilates, `Quilates <= 0.50` divide las 8 muestras en:
  - $S_{LL}$ (Quilates $\le 0.50$): IDs [1, 2] $\rightarrow$ 2 A, 0 E ($G=0.0000$, Puro $\rightarrow$ Clase A).
  - $S_{LR}$ (Quilates $> 0.50$): IDs [4, 6, 7, 9, 11, 12] $\rightarrow$ 0 A, 6 E ($G=0.0000$, Puro $\rightarrow$ Clase E).
- **Reglas del Árbol:**
  1. `SI Profundidad <= 63.5 Y Quilates <= 0.50 ENTONCES Clase = A` [Soporte: 2 A, 0 E]
  2. `SI Profundidad <= 63.5 Y Quilates > 0.50 ENTONCES Clase = E` [Soporte: 0 A, 6 E]
  3. `SI Profundidad > 63.5 ENTONCES Clase = A` [Soporte: 4 A, 0 E]

**Predicciones sobre casos nuevos:**
- **N1** (0.45, 62%): Profundidad $\le 63.5 \rightarrow$ Quilates $\le 0.50 \rightarrow$ **A** (Hoja 1)
- **N2** (1.15, 64%): Profundidad $> 63.5 \rightarrow$ **A** (Hoja 3)
- **N3** (0.85, 61%): Profundidad $\le 63.5 \rightarrow$ Quilates $> 0.50 \rightarrow$ **E** (Hoja 2)
- **N4** (0.55, 63.5%): Profundidad $\le 63.5 \rightarrow$ Quilates $> 0.50 \rightarrow$ **E** (Hoja 2)

### 4. Evaluación de Predicciones Binarias (Sección 4)
Sobre la matriz de 20 casos independientes (Real A: 6 pred A, 2 pred E; Real E: 3 pred A, 9 pred E):
- **Exactitud:** $\frac{6 + 9}{20} = \frac{15}{20} = \mathbf{0.7500 \ (75.00\%)}$
- **Precisión ($A$):** $\frac{6}{6 + 3} = \frac{6}{9} = \mathbf{0.6667 \ (66.67\%)}$
- **Exhaustividad / Recall ($A$):** $\frac{6}{6 + 2} = \frac{6}{8} = \mathbf{0.7500 \ (75.00\%)}$
- **F1-Score ($A$):** $\frac{2(6)}{2(6) + 3 + 2} = \frac{12}{17} = \mathbf{0.7059 \ (70.59\%)}$

**Desbalance 95 A / 5 E con predictor mayoritario:**
- Exactitud: $\frac{95}{100} = \mathbf{95.00\%}$
- Recall de clase E: $\frac{0}{5} = \mathbf{0.00\%}$
- Ilustra la *paradoja de la exactitud*: un modelo inútil para la clase crítica minoritaria muestra un 95% superficial. Se exige evaluar métricas por clase o Macro F1.

### 5. Verificación Scikit-Learn y Experimento Iris (Sección 5)
- **Verificación Diamantes:** Scikit-Learn reproduce exactamente la misma estructura de árbol (`Profundidad <= 63.5` en raíz, `Quilates <= 0.50` en hijo izquierdo) y coincide en un 100% en las predicciones N1-N4.
- **Iris (Partición Estratificada 60/20/20, Semilla 42):**
  - Train: 90 flores (30-30-30), Val: 30 flores (10-10-10), Test: 30 flores (10-10-10).

**Tabla Comparativa de Candidatos:**
| Candidato | Train Acc | Val Acc | Val Macro F1 | Profundidad Efectiva | Hojas |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `max_depth=1` | 0.6667 | 0.6667 | 0.5556 | 1 | 2 |
| `max_depth=3` | 0.9889 | 0.9000 | 0.8997 | 3 | 4 |
| `max_depth=None` | 1.0000 | **0.9333** | **0.9333** | 4 | 5 |

- **Modelo Seleccionado:** `max_depth=None` (mayor F1 Macro en validación).
- **Evaluación Final en Test (N=30):**
  - Exactitud Prueba: **0.9333 (93.33%)**
  - F1 Macro Prueba: **0.9333**
  - Setosa: Precisión 1.0000, Recall 1.0000, F1 1.0000
  - Versicolor: Precisión 0.9000, Recall 0.9000, F1 0.9000
  - Virginica: Precisión 0.9000, Recall 0.9000, F1 0.9000
- **Comparación contra DummyClassifier (most_frequent):**
  - Dummy Exactitud: **0.3333 (33.33%)**
  - Dummy F1 Macro: **0.1667**

---

## 📄 Entregables Generados

1. **`Informe_Taller_Arboles.pdf`**: Documento PDF de 8 páginas con diagramas, tablas comparativas, desarrollo analítico, métricas y respuestas conceptuales.
2. **`taller_arboles.ipynb`**: Cuaderno Jupyter ejecutable con el código fuente y las salidas detalladas.
3. **`figures/`**: Directorio con todas las figuras generadas en alta resolución.
