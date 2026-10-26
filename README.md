# Bocado: asistente experto en nutrición

Proyecto Final de IA Generativa. Asistente que responde preguntas sobre nutrición y alimentación saludable usando **RAG con Gemini, ChromaDB y LangGraph**.

## 1. Dominio elegido
**Nutrición y alimentación saludable.** Es un tema que le interesa a casi todo el mundo y tiene mucha información fiable. Bocado ayuda a entender conceptos (nutrientes, dieta mediterránea, etiquetas, mitos) y no sustituye a un médico ni a un dietista-nutricionista.

## 2. Base de conocimiento
Tres documentos PDF (23 páginas en total) en la carpeta `data/`:

| Documento | Contenido |
|---|---|
| `Nutricion_1_Fundamentos.pdf` | Nutrientes, energía, vitaminas, minerales y agua |
| `Nutricion_2_Alimentacion_Saludable.pdf` | Grupos de alimentos, dieta mediterránea, etiquetas, etapas de la vida |
| `Nutricion_3_Habitos_y_Mitos.pdf` | Hábitos, menús, compra, control de peso y mitos |

**Origen:** los tres documentos están escritos para este proyecto, basándose en recomendaciones generales de la OMS y de la EFSA. Son divulgativos y no son documentos oficiales.

## 3. Cómo funciona
1. Se leen los PDFs página a página (`PyPDFLoader`) y se guarda el documento y la página de cada una.
2. Se limpia el texto y se divide en 73 fragmentos de 800 caracteres con 120 de solape.
3. Cada fragmento se convierte en un vector con **Gemini Embeddings** (`gemini-embedding-2`) y se guarda en **ChromaDB**.
4. El agente, creado con **LangGraph**, tiene 3 pasos: reformular la pregunta, recuperar los 5 fragmentos más parecidos y generar la respuesta con **Gemini** (`gemini-3.5-flash-lite`).
5. La **memoria** de la conversación se guarda con `InMemorySaver` y un `thread_id` por conversación.

## 4. Decisiones técnicas
- **Fragmentos de 800 caracteres:** suficientemente pequeños para encontrar la parte exacta y suficientemente grandes para tener sentido.
- **Solape de 120:** evita cortar una idea por la mitad.
- **5 fragmentos recuperados (k=5):** con menos se escapaba información en algunas preguntas.
- **Paso de reformulación:** permite entender preguntas como "¿Y cuántas veces por semana?" usando la conversación anterior.

## 5. Prompt del sistema y justificación
El prompt completo está en la celda 4 del notebook. Define a Bocado como asistente experto en nutrición con 7 reglas.

| Técnica | Cómo se usa | Para qué |
|---|---|---|
| Persona | "Eres Bocado, un asistente experto en nutrición" | Tono y rol claros |
| Objetivo y restricciones | Objetivo + 7 reglas numeradas | Respuestas ordenadas y previsibles |
| Delimitadores | El contexto va entre `<contexto>` y `</contexto>` | Separar lo que son datos de lo que son instrucciones |
| Verificación de condición | "Si el contexto no tiene la respuesta, di…" | Evitar que se invente información |
| Filtro de tema | Rechaza preguntas que no son de nutrición | Mantener el dominio |
| Seguridad | No da diagnósticos, dietas personalizadas ni dosis | Evitar consejos médicos |
| Formato de salida | Respuestas cortas y línea de "Fuentes" | Respuestas claras y comprobables |

## 6. Ejemplos documentados
La celda 6 contiene 5 preguntas: dato concreto, concepto, mito, pregunta con memoria y pregunta fuera de tema. La celda 5 demuestra la memoria: la pregunta "¿Y cuántas veces por semana?" se entiende en una conversación con historial y no se entiende sin él.

## 7. Instalación y ejecución
**Requisitos:** Python 3.10 o superior y una clave de API de Gemini (gratuita en [Google AI Studio](https://aistudio.google.com/apikey)).
**Importante:** el proyecto no incluye ninguna clave de API. Cada persona debe usar la suya, creando un archivo `.env` como se explica a continuación.

```bash
# 1. crear y activar el entorno
python3 -m venv .venv
source .venv/bin/activate

# 2. instalar las librerías
python -m pip install -r requirements.txt

# 3. crear el archivo .env con tu clave
cp .env.example .env
# abre .env y escribe: GEMINI_API_KEY=tu_clave
```

Después abre `Bocado.ipynb`, selecciona el kernel `.venv` y pulsa **Run All**. La última celda es un chat de texto. Para terminar, escribe `salir`.

## 8. Limitaciones
- Solo conoce el contenido de los 3 PDFs.
- La memoria se pierde al reiniciar el notebook.
- Los documentos son divulgativos y no sustituyen a una fuente oficial ni a un profesional sanitario.
- A veces responde "no tengo información" a preguntas razonables que los PDFs no cubren (por ejemplo, dosis de suplementos), y es lo esperado.

## 9. Posibles mejoras
Añadir más documentos, guardar la memoria en una base de datos y crear una interfaz web con Streamlit.

## 10. Aplicación web (Streamlit)
Versión web de Bocado, con la misma lógica que el notebook (RAG, memoria con LangGraph y los mismos 3 documentos):

**URL pública:** https://bocado-asistente-nutricion.streamlit.app

El código está en `app.py`. Para ejecutarla en local:

```bash
streamlit run app.py
```