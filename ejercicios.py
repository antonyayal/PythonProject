import pandas as pd
import re
import unicodedata
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Asegurar que nltk busca en la carpeta correcta
nltk.data.path.append("nltk_data")

# Descargar stopwords si no están disponibles (Descomentar si es la primera vez)
# nltk.download("stopwords")

# Definir el idioma de stopwords (español)
stop_words = set(stopwords.words("spanish"))


def cargar_datos(ruta):
    """Carga un archivo XLSX en un DataFrame de Pandas."""
    return pd.read_excel(ruta, engine="openpyxl")


def eliminar_acentos(texto):
    """Elimina los acentos de un texto."""
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFKD', texto)
        texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return texto


def limpiar_texto(texto):
    """Convierte el texto a minúsculas, elimina acentos y puntuación."""
    texto = texto.lower()
    texto = eliminar_acentos(texto)
    texto = re.sub(r"[^\w\s]", "", texto)  # Eliminar puntuación
    return texto


def seleccionar_top_palabras(df, categoria, num_palabras=10):
    """Filtra el DataFrame por categoría y extrae las palabras más comunes."""
    df_filtrado = df[df["Category"] == categoria]
    df_muestra = df_filtrado.sample(n=10, random_state=42)
    
    texto_completo = " ".join(df_muestra["Text"].dropna())
    texto_completo = limpiar_texto(texto_completo)
    
    palabras = [word for word in texto_completo.split() if word not in stop_words]
    contador_palabras = Counter(palabras)
    
    return contador_palabras.most_common(num_palabras)


def seleccionar_top_ngrams(df, categoria, n_ngrams=4, num_top=10):
    """Filtra el DataFrame por categoría y extrae los n-grams más comunes."""
    df_filtrado = df[df["Category"] == categoria]
    df_muestra = df_filtrado.sample(n=10, random_state=42)
    
    texto_completo = " ".join(df_muestra["Text"].dropna())
    texto_completo = limpiar_texto(texto_completo)
    
    palabras = [word for word in texto_completo.split() if word not in stop_words]
    ngramas_lista = list(ngrams(palabras, n_ngrams))
    contador_ngrams = Counter(ngramas_lista)
    
    return contador_ngrams.most_common(num_top)


def graficar_top_palabras(top_palabras):
    """Genera un gráfico de barras para las palabras más comunes."""
    palabras, frec_palabras = zip(*top_palabras)
    plt.figure(figsize=(6, 5))
    plt.barh(palabras[::-1], frec_palabras[::-1])
    plt.xlabel("Frecuencia")
    plt.ylabel("Palabra")
    plt.title("Top Palabras Más Frecuentes")
    plt.show()


def graficar_top_ngrams(top_ngrams):
    """Genera un gráfico de barras para los n-grams más comunes."""
    ngramas, frec_ngrams = zip(*top_ngrams)
    ngramas = [" ".join(ngram) for ngram in ngramas]
    plt.figure(figsize=(6, 5))
    plt.barh(ngramas[::-1], frec_ngrams[::-1])
    plt.xlabel("Frecuencia")
    plt.ylabel("N-Grama")
    plt.title("Top N-Gramas Más Frecuentes")
    plt.show()


# Uso del código
ruta_archivo = "corpus/development.xlsx"  # Cambia esto por la ruta correcta
df = cargar_datos(ruta_archivo)

top_palabras = seleccionar_top_palabras(df, "Fake")
top_ngrams = seleccionar_top_ngrams(df, "Fake", n_ngrams=4)

graficar_top_palabras(top_palabras)
graficar_top_ngrams(top_ngrams)
