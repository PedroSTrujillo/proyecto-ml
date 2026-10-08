"""Carga las funciones del notebook y define los preprocesadores a comparar con regresión logística."""
import json, io, contextlib, re, os
from functools import partial
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

PROYECTO = "/Users/pedropablosanintrujillo/GitHub/proyecto-ml"
_nb = json.load(open(f"{PROYECTO}/model_training.ipynb"))
_cwd = os.getcwd(); os.chdir(PROYECTO)
for _c in _nb["cells"]:
    if _c.get("id") in ("ccbb25b2", "bb3d812a", "65a7cdfc", "b6e366f8", "4fc1af8c", "6a9db21c"):
        _src = "".join(_c["source"])
        if _c["id"] == "4fc1af8c":
            _src = _src[: _src.rindex("\nextraer_opiniones(")]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(_src, globals())
os.chdir(_cwd)

N_POSICIONES = 5
APARTE = ("por cierto", "de paso", "ademas", "por otro lado")


def oraciones_de(texto):
    return [s.strip(" ,") for s in re.split(r"[.!?]", texto) if s.strip(" ,")]


def agregar_columnas(X):
    """Agrega todas las columnas derivadas que usan los distintos preprocesadores."""
    t = X[TEXT_COLUMN]
    ors = t.map(oraciones_de)
    cols = {
        "ultima": ors.map(lambda o: o[-1] if o else ""),
        "contraste": t.map(extract_after_contrast),
        "sin_primera": ors.map(lambda o: ". ".join(o[1:])),
        "decisiva": ors.map(lambda o: next((s for s in reversed(o) if PATRON_ASPECTO.search(quitar_tildes(s.lower()))
                                            and not quitar_tildes(s.lower()).startswith(APARTE)), "")),
    }
    for k in range(N_POSICIONES):
        cols[f"ini_{k}"] = ors.map(lambda o, k=k: o[k] if len(o) > k else "")
        cols[f"fin_{k}"] = ors.map(lambda o, k=k: o[-1 - k] if len(o) > k else "")
    return X.assign(**cols)


def palabras(ngram=(1, 2), stemming=True, sublinear=False, min_df=1):
    return TfidfVectorizer(tokenizer=partial(tokenizer_stemmer, stemming=stemming), token_pattern=None,
                           ngram_range=ngram, sublinear_tf=sublinear, min_df=min_df)


def caracteres(ngram=(2, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, sublinear_tf=True, min_df=2)


def armar(vectorizadores):
    """Pipeline sklearn: columnas derivadas + un vectorizador por columna."""
    return Pipeline([("columnas", FunctionTransformer(agregar_columnas)),
                     ("vectorizadores", ColumnTransformer(vectorizadores))])


PREPROCESADORES = {
    # Texto completo
    "P1 palabras(1,2) texto": lambda: armar([("t", palabras(), "text")]),
    "P2 palabras(1,3) texto": lambda: armar([("t", palabras((1, 3)), "text")]),
    "P3 palabras(1,2) sin stemming": lambda: armar([("t", palabras(stemming=False), "text")]),
    "P4 caracteres(2,5) texto": lambda: armar([("t", caracteres(), "text")]),
    "P5 caracteres(3,6) texto": lambda: armar([("t", caracteres((3, 6)), "text")]),
    # Partes del texto por separado
    "P6 solo última oración": lambda: armar([("u", palabras(), "ultima")]),
    "P7 solo después del contraste": lambda: armar([("c", palabras(), "contraste")]),
    "P8 solo oración decisiva": lambda: armar([("d", palabras(), "decisiva")]),
    # Combinaciones de columnas
    "P9 actual (sin 1ª + última + contraste)": lambda: armar([("t", palabras(), "sin_primera"), ("u", palabras(), "ultima"), ("c", palabras(), "contraste")]),
    "P10 actual + decisiva": lambda: armar([("t", palabras(), "sin_primera"), ("u", palabras(), "ultima"), ("c", palabras(), "contraste"), ("d", palabras(), "decisiva")]),
    "P11 actual + caracteres": lambda: armar([("t", palabras(), "sin_primera"), ("u", palabras(), "ultima"), ("c", palabras(), "contraste"), ("ch", caracteres(), "text")]),
    "P12 caracteres en última + contraste + texto": lambda: armar([("t", caracteres(), "text"), ("u", caracteres(), "ultima"), ("c", caracteres(), "contraste")]),
    # Cada oración en su propia columna
    "P13 oraciones por posición desde el inicio": lambda: armar([(f"i{k}", palabras(), f"ini_{k}") for k in range(N_POSICIONES)]),
    "P14 oraciones por posición desde el final": lambda: armar([(f"f{k}", palabras(), f"fin_{k}") for k in range(N_POSICIONES)]),
    "P15 posiciones desde el final + texto": lambda: armar([(f"f{k}", palabras(), f"fin_{k}") for k in range(N_POSICIONES)] + [("t", palabras(), "text")]),
}
