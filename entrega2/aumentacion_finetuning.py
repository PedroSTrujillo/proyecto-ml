import argparse
import gc
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


parser = argparse.ArgumentParser()
parser.add_argument("--prueba", action="store_true", help="Corrida corta para verificar que todo funciona")
args = parser.parse_args()

SEMILLA = 42
CARPETA_DATOS = "./datos"
TRAIN_PARTE_1 = "../entrega1/data/train.csv"
CARPETA_SALIDA = "./resultados_aumentacion" + ("_prueba" if args.prueba else "")

# Modelos base (Hugging Face). Se pueden cambiar con variables de entorno, sin tocar el código.
MODELO_MBERT = os.environ.get("MODELO_MBERT", "bert-base-multilingual-cased")
MODELO_XLMR = os.environ.get("MODELO_XLMR", "FacebookAI/xlm-roberta-base")

MAX_TOKENS = 256
TAMANO_LOTE = 16
TASA_APRENDIZAJE = 2e-5
MAX_EPOCAS = 1 if args.prueba else 3
PACIENCIA = 1

URL_DICCIONARIO = "https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt"
RUTA_DICCIONARIO = os.path.join(CARPETA_DATOS, "diccionario_es-en_MUSE.txt")

# Aumentación
N_CODE_SWITCHING = 3000      # reseñas en español que se convierten en reseñas mezcladas español-inglés
PROB_TRADUCIR = 0.3          # probabilidad de traducir cada palabra que esté en el diccionario
PROB_ERROR_PALABRA = 0.1     # probabilidad de meterle un error de escritura a cada palabra
IDIOMAS_CON_ERRORES = ("en", "pt_br", "fr")  # idiomas con menos reseñas en train que en eval

os.makedirs(CARPETA_SALIDA, exist_ok=True)


def fijar_semillas(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


fijar_semillas()
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else ""))
if dispositivo.type == "cpu" and not args.prueba:
    print("ADVERTENCIA: no se detectó GPU. La corrida completa en CPU puede tardar muchísimo.")

dataset_train = pd.read_csv(os.path.join(CARPETA_DATOS, "train.csv"))
dataset_eval = pd.read_csv(os.path.join(CARPETA_DATOS, "eval.csv"))

puntua = {i: " " for i in range(sys.maxunicode + 1) if unicodedata.category(chr(i)).startswith("P")}
AMBIGUAS_EN = {"he"}


def procesar_resena(text):
    return text.lower().translate(puntua).split()


def palabras_exclusivas(palabras_por_idioma):
    return {
        idioma: palabras - set().union(*(otras for otro, otras in palabras_por_idioma.items() if otro != idioma))
        for idioma, palabras in palabras_por_idioma.items()
    }


def identificar_idioma(resena, palabras_por_idioma, exclusivas):
    conteo = Counter(procesar_resena(resena))

    def puntajes(listas):
        return {idioma: sum(n for palabra, n in conteo.items() if palabra in palabras) for idioma, palabras in listas.items()}

    puntaje = puntajes(exclusivas)
    if max(puntaje.values()) == 0:
        puntaje = puntajes(palabras_por_idioma)
        if max(puntaje.values()) == 0:
            return None
    principal = max(puntaje, key=lambda k: puntaje[k])
    hay_es = any(palabra in exclusivas["es"] for palabra in conteo)
    hay_en = any(palabra in exclusivas["en"] - AMBIGUAS_EN for palabra in conteo)
    if principal in ("es", "en") and hay_es and hay_en:
        return "es_en"
    return principal


palabras_comunes = pd.read_csv(os.path.join(CARPETA_DATOS, "50_palabras_mas_comunes", "50_palabras_mas_comunes.csv"))
palabras_comunes["palabra"] = palabras_comunes["palabra"].map(lambda p: procesar_resena(p)[0])
palabras_por_idioma = palabras_comunes.groupby("idioma")["palabra"].apply(set).to_dict()
exclusivas = palabras_exclusivas(palabras_por_idioma)

dataset_train["idioma"] = dataset_train["text"].map(lambda t: identificar_idioma(t, palabras_por_idioma, exclusivas))
dataset_eval["idioma"] = dataset_eval["text"].map(lambda t: identificar_idioma(t, palabras_por_idioma, exclusivas))

# Reseñas que ya estaban en la parte 1 (español con plantilla) vs. reseñas nuevas de la parte 2
textos_parte_1 = set(pd.read_csv(TRAIN_PARTE_1)["text"]) if os.path.exists(TRAIN_PARTE_1) else set()
dataset_train["origen"] = np.where(dataset_train["text"].isin(textos_parte_1), "parte_1", "nueva")

etiquetas = sorted(dataset_train["label"].unique())  # ['negativo', 'neutral', 'positivo']
etiqueta_a_id = {e: i for i, e in enumerate(etiquetas)}

# División 90/10 estratificada por idioma y etiqueta (idéntica a la del notebook del grupo)
estrato = dataset_train["idioma"].astype(str) + "|" + dataset_train["label"]
raros = estrato.map(estrato.value_counts()) < 2
train_df, test_df = train_test_split(dataset_train[~raros], test_size=0.1, random_state=SEMILLA, stratify=estrato[~raros])
train_df = pd.concat([train_df, dataset_train[raros]])

if args.prueba:
    train_df = train_df.sample(600, random_state=SEMILLA)
    test_df = test_df.sample(200, random_state=SEMILLA)
    dataset_eval = dataset_eval.head(200)
    N_CODE_SWITCHING = 100

print(f"Entrenamiento: {len(train_df):,} | Validación: {len(test_df):,} | Eval: {len(dataset_eval):,}")
print("Idiomas en validación:", test_df["idioma"].value_counts().to_dict())


def cargar_diccionario():
    """Diccionario español->inglés de MUSE. Se descarga una sola vez y se guarda en ./datos."""
    if not os.path.exists(RUTA_DICCIONARIO):
        print(f"Descargando diccionario bilingüe de {URL_DICCIONARIO} ...")
        urllib.request.urlretrieve(URL_DICCIONARIO, RUTA_DICCIONARIO)
    traducciones = {}
    with open(RUTA_DICCIONARIO, encoding="utf-8") as archivo:
        for linea in archivo:
            partes = linea.split()
            # Nos quedamos con la primera traducción de cada palabra y descartamos las que son iguales
            if len(partes) == 2 and partes[0] != partes[1] and partes[0] not in traducciones:
                traducciones[partes[0]] = partes[1]
    return traducciones


def code_switching(texto, traducciones, rng):
    """Reemplaza al azar palabras en español por su traducción al inglés (conserva mayúscula inicial y puntuación)."""
    def reemplazar(m):
        palabra = m.group(0)
        traduccion = traducciones.get(palabra.lower())
        if traduccion is None or rng.random() > PROB_TRADUCIR:
            return palabra
        return traduccion.capitalize() if palabra[0].isupper() else traduccion

    return re.sub(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", reemplazar, texto)


def error_de_escritura(palabra, rng):
    """Un error típico de tecleo: borrar, duplicar o intercambiar una letra, o quitar una tilde."""
    if len(palabra) < 4:
        return palabra
    i = rng.randrange(1, len(palabra) - 1)
    tipo = rng.choice(["borrar", "duplicar", "intercambiar", "tilde"])
    if tipo == "borrar":
        return palabra[:i] + palabra[i + 1:]
    if tipo == "duplicar":
        return palabra[:i] + palabra[i] + palabra[i:]
    if tipo == "intercambiar":
        return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]
    sin_tildes = unicodedata.normalize("NFKD", palabra).encode("ascii", "ignore").decode("ascii")
    return sin_tildes if sin_tildes else palabra


def meter_errores(texto, rng):
    return re.sub(
        r"\w+",
        lambda m: error_de_escritura(m.group(0), rng) if rng.random() < PROB_ERROR_PALABRA else m.group(0),
        texto,
    )


def aumentar(train_df):
    """Devuelve train_df + reseñas nuevas generadas. Las etiquetas no cambian."""
    rng = random.Random(SEMILLA)

    try:
        traducciones = cargar_diccionario()
        print(f"Diccionario bilingüe: {len(traducciones):,} palabras")
        espanol = train_df[train_df["idioma"] == "es"].sample(N_CODE_SWITCHING, random_state=SEMILLA)
        mezcladas = espanol.assign(
            text=espanol["text"].map(lambda t: code_switching(t, traducciones, rng)),
            idioma="es_en_sintetico",
            aumentacion="code_switching",
        )
    except Exception as error:
        print(f"ADVERTENCIA: no se pudo usar el diccionario bilingüe ({error!r}). "
              "Se sigue solo con los errores de escritura.")
        mezcladas = train_df.iloc[0:0].assign(aumentacion="code_switching")

    otros = train_df[train_df["idioma"].isin(IDIOMAS_CON_ERRORES)]
    con_errores = otros.assign(
        text=otros["text"].map(lambda t: meter_errores(t, rng)),
        aumentacion="errores",
    )

    nuevas = pd.concat([mezcladas, con_errores])
    nuevas = nuevas[nuevas["text"] != train_df.loc[nuevas.index, "text"].values]  # descarta copias idénticas

    # Ejemplos notebook
    ejemplos = pd.DataFrame(
        {
            "tipo": nuevas["aumentacion"].values,
            "original": train_df.loc[nuevas.index, "text"].values,
            "aumentada": nuevas["text"].values,
            "label": nuevas["label"].values,
        }
    )
    ejemplos.groupby("tipo").head(15).to_csv(os.path.join(CARPETA_SALIDA, "ejemplos_aumentacion.csv"), index=False)

    print(f"Aumentación: +{(nuevas['aumentacion'] == 'code_switching').sum():,} code-switching, "
          f"+{(nuevas['aumentacion'] == 'errores').sum():,} con errores de escritura")
    return pd.concat([train_df.assign(aumentacion="original"), nuevas], ignore_index=True)


train_df_aumentado = aumentar(train_df)
print(f"Entrenamiento con aumentación: {len(train_df_aumentado):,} reseñas")


def crear_loader(df, tokenizador, shuffle):
    ejemplos = list(zip(df["text"], df["label"].map(etiqueta_a_id)))

    def armar_lote(lote):
        textos, ids = zip(*lote)
        entradas = tokenizador(list(textos), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        entradas["labels"] = torch.tensor(ids)
        return entradas

    generador = torch.Generator().manual_seed(SEMILLA)
    return DataLoader(ejemplos, batch_size=TAMANO_LOTE, shuffle=shuffle, collate_fn=armar_lote, generator=generador)


def usar_precision_mixta():
    return dispositivo.type == "cuda" and torch.cuda.is_bf16_supported()


@torch.no_grad()
def predecir(model, loader):
    """Pérdida promedio y clases predichas (si el loader trae etiquetas)."""
    model.eval()
    perdida, n, clases = 0.0, 0, []
    for lote in loader:
        lote = lote.to(dispositivo)
        with torch.autocast(device_type=dispositivo.type, dtype=torch.bfloat16, enabled=usar_precision_mixta()):
            salida = model(**lote)
        if "labels" in lote:
            perdida += salida.loss.float().item() * len(lote["labels"])
            n += len(lote["labels"])
        clases.append(salida.logits.float().argmax(dim=-1).cpu())
    return (perdida / n if n else None), torch.cat(clases).numpy()


def entrenar(nombre_modelo, df_train, tokenizador):
    fijar_semillas()
    model = AutoModelForSequenceClassification.from_pretrained(
        nombre_modelo, num_labels=len(etiquetas), id2label=dict(enumerate(etiquetas)), label2id=etiqueta_a_id
    ).to(dispositivo)

    train_loader = crear_loader(df_train, tokenizador, shuffle=True)
    val_loader = crear_loader(test_df, tokenizador, shuffle=False)
    y_val = test_df["label"].map(etiqueta_a_id).to_numpy()

    optimizador = torch.optim.AdamW(model.parameters(), lr=TASA_APRENDIZAJE)
    pasos = MAX_EPOCAS * len(train_loader)
    planificador = get_linear_schedule_with_warmup(optimizador, int(0.1 * pasos), pasos)

    historial, mejor_epoca, mejor_estado = [], 0, None
    for epoca in range(MAX_EPOCAS):
        model.train()
        perdida, aciertos, n = 0.0, 0, 0
        t0 = time.time()
        for lote in tqdm(train_loader, desc=f"Época {epoca + 1}/{MAX_EPOCAS}", mininterval=10):
            lote = lote.to(dispositivo)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.bfloat16, enabled=usar_precision_mixta()):
                salida = model(**lote)
            salida.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizador.step()
            planificador.step()
            optimizador.zero_grad()
            perdida += salida.loss.float().item() * len(lote["labels"])
            aciertos += (salida.logits.argmax(dim=-1) == lote["labels"]).sum().item()
            n += len(lote["labels"])

        val_loss, pred_val = predecir(model, val_loader)
        fila = {
            "epoca": epoca + 1,
            "loss": perdida / n,
            "accuracy": aciertos / n,
            "val_loss": val_loss,
            "val_accuracy": (pred_val == y_val).mean(),
            "minutos": (time.time() - t0) / 60,
        }
        historial.append(fila)
        print({k: round(v, 4) for k, v in fila.items()})

        if val_loss <= min(h["val_loss"] for h in historial):
            mejor_epoca = epoca
            mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoca - mejor_epoca >= PACIENCIA:
            print(f"Parada temprana en la época {epoca + 1}")
            break

    model.load_state_dict(mejor_estado)
    print(f"Mejor época según val_loss: {mejor_epoca + 1}")
    return model, pd.DataFrame(historial), mejor_epoca


def correr(nombre_corrida, nombre_modelo, df_train):
    print(f"\n{'=' * 90}\nCorrida {nombre_corrida}: {nombre_modelo} | {len(df_train):,} reseñas de entrenamiento\n{'=' * 90}")
    t0 = time.time()
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    model, historial, mejor_epoca = entrenar(nombre_modelo, df_train, tokenizador)
    historial.to_csv(os.path.join(CARPETA_SALIDA, f"historial_{nombre_corrida}.csv"), index=False)

    _, pred_val = predecir(model, crear_loader(test_df, tokenizador, shuffle=False))
    analisis = test_df[["id", "text", "idioma", "origen", "label"]].copy()
    analisis["pred"] = [etiquetas[c] for c in pred_val]
    analisis["acierto"] = analisis["pred"] == analisis["label"]
    analisis.to_csv(os.path.join(CARPETA_SALIDA, f"predicciones_val_{nombre_corrida}.csv"), index=False)

    resultado = {
        "corrida": nombre_corrida,
        "modelo": nombre_modelo,
        "reseñas_train": len(df_train),
        "mejor_epoca": mejor_epoca + 1,
        "val_loss": historial["val_loss"].iloc[mejor_epoca],
        "val_accuracy": analisis["acierto"].mean(),
        **{f"acc_{idioma}": grupo["acierto"].mean() for idioma, grupo in analisis.groupby("idioma")},
        **{f"acc_{origen}": grupo["acierto"].mean() for origen, grupo in analisis.groupby("origen")},
        "minutos": (time.time() - t0) / 60,
    }

    eval_loader = DataLoader(
        list(dataset_eval["text"]),
        batch_size=64,
        collate_fn=lambda textos: tokenizador(list(textos), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"),
    )
    _, pred_eval = predecir(model, eval_loader)
    submission = pd.DataFrame({"id": dataset_eval["id"], "answer": [etiquetas[c] for c in pred_eval]})
    submission.to_csv(os.path.join(CARPETA_SALIDA, f"submission_{nombre_corrida}.csv"), index=False)
    print("Distribución de predicciones en eval:", submission["answer"].value_counts(normalize=True).round(3).to_dict())

    # Modelo entrenado
    carpeta_modelo = os.path.join(CARPETA_SALIDA, f"modelo_{nombre_corrida}")
    model.save_pretrained(carpeta_modelo)
    tokenizador.save_pretrained(carpeta_modelo)

    ruta_resumen = os.path.join(CARPETA_SALIDA, "resumen.csv")
    resumen = pd.read_csv(ruta_resumen) if os.path.exists(ruta_resumen) else pd.DataFrame()
    resumen = pd.concat([resumen[resumen.get("corrida", pd.Series(dtype=str)) != nombre_corrida], pd.DataFrame([resultado])])
    resumen.to_csv(ruta_resumen, index=False)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in resultado.items()}, indent=2, ensure_ascii=False))

    del model
    gc.collect()
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    return resultado


resultados = {}
for nombre, modelo, datos in [
    ("A_mbert_sin_aumentacion", MODELO_MBERT, train_df),
    ("B_mbert_con_aumentacion", MODELO_MBERT, train_df_aumentado),
]:
    try:
        resultados[nombre] = correr(nombre, modelo, datos)
    except Exception as error:  # una falla no debe tumbar las demás corridas
        print(f"ERROR en la corrida {nombre}: {error!r}")

if resultados:
    con_aumentacion = resultados.get("B_mbert_con_aumentacion", {}).get("val_accuracy", 0) > resultados.get(
        "A_mbert_sin_aumentacion", {}
    ).get("val_accuracy", 0)
    nombre_c = "C_xlmr_" + ("con" if con_aumentacion else "sin") + "_aumentacion"
    try:
        correr(nombre_c, MODELO_XLMR, train_df_aumentado if con_aumentacion else train_df)
    except Exception as error:
        print(f"ERROR en la corrida {nombre_c}: {error!r}")

print(f"\nListo. Resultados en {os.path.abspath(CARPETA_SALIDA)}")
ruta_resumen = os.path.join(CARPETA_SALIDA, "resumen.csv")
if os.path.exists(ruta_resumen):
    print(pd.read_csv(ruta_resumen).round(4).to_string(index=False))
