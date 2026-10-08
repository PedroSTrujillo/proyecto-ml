import pickle, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import prep

R = pickle.load(open("oof.pkl", "rb"))
oof, y, clases, CS = R["oof"], R["y"], R["clases"], R["CS"]
textos = prep.data.loc[R["index"], "text"]
ci = {c: i for i, c in enumerate(clases)}

# Tipo de reseña según el extractor de opiniones del notebook
DECISIVOS = {"al final", "aun asi", "con todo", "pero", "a fin de cuentas", "sin embargo", "aunque"}
def tipo(texto, etiqueta):
    o = prep.extraer_opiniones(texto)
    if etiqueta == "neutral": return "neutral"
    if not o: return "sin opinión detectada"
    if len({p for _, p, _ in o}) == 1: return "una polaridad"
    return "mixta, cierre decisivo" if o[-1][2] in DECISIVOS else "mixta, sin cierre decisivo"
tipos = np.array([tipo(t, l) for t, l in zip(textos, y)])

def metricas(P):
    pred = np.array(clases)[P.argmax(1)]
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=clases, zero_division=0)
    nn = y != "neutral"
    pn = P[nn][:, [ci["negativo"], ci["positivo"]]]
    sep = (np.array(["negativo", "positivo"])[pn.argmax(1)] == y[nn]).mean()
    m = {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro")}
    m |= {f"f1_{c}": f for c, f in zip(clases, f1)} | {f"recall_{c}": r for c, r in zip(clases, rc)}
    m["separa_pos_neg"] = sep
    for g in ["neutral", "una polaridad", "mixta, cierre decisivo", "mixta, sin cierre decisivo", "sin opinión detectada"]:
        m[f"acc[{g}]"] = (pred[tipos == g] == y[tipos == g]).mean()
    return m, pred

filas, mejores, preds = [], {}, {}
for nombre, d in oof.items():
    accs = {C: accuracy_score(y, np.array(clases)[d[C].argmax(1)]) for C in CS}
    C = max(accs, key=accs.get)
    m, pred = metricas(d[C])
    mejores[nombre], preds[nombre] = d[C], pred
    filas.append({"modelo": nombre, "C": C, "n_features": d["n_feat"], "train_acc": np.mean(d["train_acc"][C])} | m)
tabla = pd.DataFrame(filas).set_index("modelo").sort_values("accuracy", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("Tamaño de cada grupo:", pd.Series(tipos).value_counts().to_dict())
print(tabla.round(4).to_string())
tabla.to_csv("tabla_modelos.csv")

# Mejor modelo por métrica
print("\nMejor modelo por métrica:")
for col in [c for c in tabla.columns if c.startswith(("f1_", "recall_", "separa", "acc["))]:
    print(f"  {col:38s} {tabla[col].idxmax():45s} {tabla[col].max():.4f}   (peor: {tabla[col].min():.4f})")

# Correlación de errores (fracción de errores compartidos, Jaccard)
nombres = list(tabla.index)
err = {n: preds[n] != y for n in nombres}
J = pd.DataFrame([[ (err[a] & err[b]).sum() / max((err[a] | err[b]).sum(), 1) for b in nombres] for a in nombres],
                 index=[n.split()[0] for n in nombres], columns=[n.split()[0] for n in nombres])
print("\nErrores compartidos (Jaccard; 1 = se equivocan en exactamente las mismas reseñas):")
print(J.round(2).to_string())
J.to_csv("errores_compartidos.csv")

# Oráculo: cota superior de cualquier combinación (alguno acierta)
print("\nOráculo (al menos un modelo acierta):", np.mean(np.any([preds[n] == y for n in nombres], axis=0)).round(4))

# Votación suave: selección voraz hacia adelante
def acc_voto(sel): return accuracy_score(y, np.array(clases)[np.mean([mejores[n] for n in sel], axis=0).argmax(1)])
sel = [nombres[0]]; actual = acc_voto(sel); hist = [(list(sel), actual)]
while True:
    cand = [(acc_voto(sel + [n]), n) for n in nombres if n not in sel]
    if not cand: break
    a, n = max(cand)
    if a <= actual: break
    sel.append(n); actual = a; hist.append((list(sel), a))
print("\nVotación suave, selección voraz:")
for s, a in hist: print(f"  {a:.4f}  {' | '.join(x.split()[0] for x in s)}")
m, _ = metricas(np.mean([mejores[n] for n in sel], axis=0))
print("  Métricas de la mejor votación:", {k: round(v, 4) for k, v in m.items()})
pickle.dump({"tabla": tabla, "J": J, "hist": hist, "mejor_voto": m, "tipos": tipos}, open("analisis.pkl", "wb"))
