import sys, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import prep

CS = [1.0, 10.0, 30.0]
X = prep.data.drop(columns=prep.TARGET_COLUMN); y = prep.data[prep.TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))

def tarea(nombre, k):
    import warnings; warnings.filterwarnings("ignore")
    import prep
    tr, te = folds[k]
    p = prep.PREPROCESADORES[nombre]()
    A = p.fit_transform(X_train.iloc[tr]); B = p.transform(X_train.iloc[te])
    out = {}
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000).fit(A, y_train.iloc[tr])
        out[C] = (m.predict_proba(B), m.score(A, y_train.iloc[tr]))
    return nombre, k, out, A.shape[1], list(m.classes_)

t = time.time()
res = Parallel(n_jobs=-1, verbose=0)(delayed(tarea)(n, k) for n in prep.PREPROCESADORES for k in range(5))
oof = {}
for nombre, k, out, nfeat, clases in res:
    d = oof.setdefault(nombre, {C: np.zeros((len(X_train), 3)) for C in CS} | {"train_acc": {C: [] for C in CS}, "n_feat": nfeat})
    for C, (proba, tracc) in out.items():
        d[C][folds[k][1]] = proba; d["train_acc"][C].append(tracc)
pickle.dump({"oof": oof, "y": y_train.values, "clases": clases, "CS": CS, "index": X_train.index.values},
            open("oof.pkl", "wb"))
print(f"listo en {time.time()-t:.0f}s")
