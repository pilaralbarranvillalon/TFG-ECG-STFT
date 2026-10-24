"""Evaluación final de los modelos elegidos en arquitecturas_tfg.py.

Carga los pesos ya entrenados (no vuelve a entrenar) y genera en outputs/evaluacion_final:
matrices de confusión 5×5, métricas por superclase, curvas ROC, curvas de pérdida del modelo final
y el diagnóstico de ECG concretos del conjunto de prueba.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import roc_curve, roc_auc_score

import experimentos_tfg as X
import arquitecturas_tfg as A

C = X.SUPER_CLASSES
NOMBRES = [X.NOMBRES[c] for c in C]
FINALES = {"senal": "A8_senal_s42", "espectrograma": "A8_espectrograma_s42"}
TITULOS = {"senal": "ResNet 1D sobre la señal temporal", "espectrograma": "Codificador por derivación sobre el espectrograma"}
SALIDA = Path("./outputs/evaluacion_final")
AZUL, NARANJA, GRIS = "#1F5A7A", "#E07B2A", "#9AA0A6"


def configurar(data_dir):
    A.configurar(data_dir)
    SALIDA.mkdir(parents=True, exist_ok=True)


def fila(id_):
    r = A.resultados()
    return r[r["id"] == id_].iloc[0]


def origen(f):
    o = f["reutilizado_de"]
    return o if isinstance(o, str) and o else f["id"]


def cargar_modelo(id_):
    f = fila(id_)
    cfg = A.cfg_de(f)
    A.R["l2"], A.R["drop"] = cfg["l2"], cfg["drop"]
    rep, _, constructor = A.ARQUITECTURAS[cfg["arquitectura"]]
    forma = (1000, 12) if rep == "senal" else (*X.SPEC_SHAPE, 12)
    modelo = constructor(forma, len(C))
    modelo.load_weights(A.OUT_DIR / "modelos" / f"{origen(f)}.weights.h5")
    return modelo, rep, np.array(json.loads(f["umbrales"])), origen(f)


def representar(senal_limpia, rep):
    return senal_limpia if rep == "senal" else X.ecg_to_tensor(senal_limpia)


def conjunto_prueba(rep):
    _, _, te = X.cargar_etiquetas()
    f = X.senal_limpia if rep == "senal" else X.espectrograma
    Xs, Ys, ids, rutas = [], [], [], []
    for _, r in te.iterrows():
        t = f(r["filename_lr"])
        if t is None:
            continue
        Xs.append(t); Ys.append(r["target_vec"]); ids.append(int(r["ecg_id"])); rutas.append(r["filename_lr"])
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32), ids, rutas


def metricas_por_clase(Y, P, th):
    pred = (P >= th).astype(int)
    filas = []
    for i, c in enumerate(C):
        y, p = Y[:, i], pred[:, i]
        tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        sens, esp = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-8)
        filas.append(dict(superclase=X.NOMBRES[c], casos=int(y.sum()), umbral=round(float(th[i]), 3),
                          VP=tp, FN=fn, FP=fp, VN=tn, sensibilidad=sens, especificidad=esp,
                          precision=prec, f1=f1, auc=roc_auc_score(y, P[:, i])))
    return pd.DataFrame(filas)


def matriz_multietiqueta(Y, pred):
    """Fila i: de los ECG que tienen la superclase i, porcentaje que recibe cada etiqueta j."""
    return np.array([pred[Y[:, i] == 1].mean(axis=0) for i in range(len(C))])


def matriz_una_etiqueta(Y, P):
    """Solo ECG con una única superclase real; la predicción es la superclase más probable."""
    una = Y.sum(axis=1) == 1
    real, pred = Y[una].argmax(axis=1), P[una].argmax(axis=1)
    M = np.zeros((len(C), len(C)), dtype=int)
    for r, p in zip(real, pred):
        M[r, p] += 1
    return M, int(una.sum())


def dibujar_matriz(M, titulo, fichero, porcentaje=True, pie=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    V = M / M.sum(axis=1, keepdims=True) if not porcentaje else M
    ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(C)):
        for j in range(len(C)):
            txt = f"{100 * V[i, j]:.0f} %" if porcentaje else f"{M[i, j]}\n({100 * V[i, j]:.0f} %)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if V[i, j] > 0.55 else "#222")
    ax.set_xticks(range(len(C))); ax.set_xticklabels(NOMBRES, rotation=30, ha="right")
    ax.set_yticks(range(len(C))); ax.set_yticklabels(NOMBRES)
    ax.set_xlabel("Superclase predicha"); ax.set_ylabel("Superclase real")
    ax.set_title(titulo, fontsize=10, loc="left")
    if pie:
        fig.text(0.01, 0.01, pie, fontsize=8, color="#555")
    fig.tight_layout(); fig.savefig(SALIDA / fichero, dpi=220); plt.close(fig)


def dibujar_roc(Y, P, rep):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    colores = ["#1F5A7A", "#E07B2A", "#4A8A6A", "#B22222", "#7A5A9E"]
    for i, c in enumerate(C):
        fpr, tpr, _ = roc_curve(Y[:, i], P[:, i])
        ax.plot(fpr, tpr, lw=2, color=colores[i], label=f"{X.NOMBRES[c]} (AUC {roc_auc_score(Y[:, i], P[:, i]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRIS, lw=1)
    ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad")
    ax.set_title(f"Curvas ROC en el conjunto de prueba: {TITULOS[rep]}", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(SALIDA / f"roc_{rep}.png", dpi=220); plt.close(fig)


def dibujar_curvas_finales():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, rep in zip(axs, ["espectrograma", "senal"]):
        f = fila(FINALES[rep])
        h = pd.read_csv(A.OUT_DIR / "historiales" / f"{origen(f)}.csv")
        ax.plot(h["epoca"], h["perdida_entren_eval"], color=GRIS, lw=2, label="Entrenamiento")
        ax.plot(h["epoca"], h["perdida_val_eval"], color=AZUL, lw=2, label="Validación")
        ax.axvline(int(f["mejor_epoca"]), color=NARANJA, ls="--", lw=1.3, label="Época restaurada")
        ax.set_title(TITULOS[rep], fontsize=10, loc="left")
        ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(SALIDA / "curvas_modelos_finales.png", dpi=220); plt.close(fig)


def diagnosticar(ruta_wfdb, modelo, rep, th, titulo="", reales=None, fichero=None):
    """Diagnostica un ECG a partir del registro WFDB sin procesar (ruta sin extensión)."""
    bruta, _ = wfdb.rdsamp(str(ruta_wfdb))
    limpia = X.preprocess_ecg(bruta)
    if limpia is None:
        print("Registro no válido: más de la mitad de las derivaciones son planas o defectuosas.")
        return None
    p = modelo.predict(representar(limpia, rep)[None], verbose=0)[0]
    tabla = pd.DataFrame(dict(superclase=NOMBRES, probabilidad=p.round(3), umbral=th.round(3),
                              diagnostico=np.where(p >= th, "Sí", "No")))
    if reales is not None:
        tabla["real"] = np.where(np.array(reales) == 1, "Sí", "No")
    print(titulo); print(tabla.to_string(index=False))
    if fichero:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
        t = np.arange(bruta.shape[0]) / X.FS
        axs[0].plot(t, bruta[:, 1], color=GRIS, lw=1, label="Señal original")
        axs[0].plot(t, limpia[:, 1] * np.std(bruta[:, 1]) + np.mean(bruta[:, 1]), color=AZUL, lw=1.2, label="Señal filtrada")
        axs[0].set_xlabel("Tiempo (s)"); axs[0].set_title(f"{titulo}: derivación II", fontsize=10, loc="left")
        axs[0].legend(frameon=False, fontsize=8)
        col = [NARANJA if s == "Sí" else "#C9D6DF" for s in tabla["diagnostico"]]
        axs[1].barh(NOMBRES[::-1], p[::-1], color=col[::-1])
        for k, u in enumerate(th[::-1]):
            axs[1].plot([u, u], [k - 0.4, k + 0.4], color="#222", lw=1.5)
        axs[1].set_xlim(0, 1); axs[1].set_xlabel("Probabilidad (la raya negra es el umbral)")
        axs[1].set_title("Diagnóstico del modelo", fontsize=10, loc="left")
        for ax in axs:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(SALIDA / fichero, dpi=220); plt.close(fig)
    return tabla


def ejecutar_todo(ejemplos=3):
    resumen = []
    for rep in ["senal", "espectrograma"]:
        modelo, rep_, th, org = cargar_modelo(FINALES[rep])
        Xte, Yte, ids, rutas = conjunto_prueba(rep)
        P = modelo.predict(Xte, verbose=0, batch_size=256)
        pred = (P >= th).astype(int)
        m = X.metricas(Yte, P, th)
        print(f"{TITULOS[rep]} ({org}): AUC {m['auc']:.4f} | F1 {m['f1']:.4f} | {len(Yte)} ECG de prueba")
        resumen.append(dict(representacion=rep, modelo=org, parametros=int(modelo.count_params()), **m))

        pc = metricas_por_clase(Yte, P, th)
        pc.to_csv(SALIDA / f"metricas_por_clase_{rep}.csv", index=False)
        M = matriz_multietiqueta(Yte, pred)
        pd.DataFrame(M, index=NOMBRES, columns=NOMBRES).to_csv(SALIDA / f"matriz_multietiqueta_{rep}.csv")
        dibujar_matriz(M, f"Matriz de confusión multietiqueta: {TITULOS[rep]}", f"matriz_multietiqueta_{rep}.png",
                       pie="Fila: ECG que tienen esa superclase. Columna: porcentaje de ellos que recibe cada etiqueta.")
        M1, n1 = matriz_una_etiqueta(Yte, P)
        pd.DataFrame(M1, index=NOMBRES, columns=NOMBRES).to_csv(SALIDA / f"matriz_una_etiqueta_{rep}.csv")
        dibujar_matriz(M1, f"Matriz de confusión 5×5: {TITULOS[rep]}", f"matriz_5x5_{rep}.png", porcentaje=False,
                       pie=f"{n1} ECG de prueba con una sola superclase; predicción = superclase más probable.")
        dibujar_roc(Yte, P, rep)
        pd.DataFrame(dict(ecg_id=ids, **{f"real_{c}": Yte[:, i] for i, c in enumerate(C)},
                          **{f"prob_{c}": P[:, i] for i, c in enumerate(C)})).to_csv(SALIDA / f"predicciones_{rep}.csv", index=False)

        if rep == "senal":
            aciertos = np.where((pred == Yte).all(axis=1) & (Yte.sum(axis=1) >= 2))[0][:ejemplos - 1]
            fallo_hyp = np.where((Yte[:, C.index("HYP")] == 1) & (pred[:, C.index("HYP")] == 0))[0][:1]
            for k in list(aciertos) + list(fallo_hyp):
                diagnosticar(X.DATA_DIR / rutas[k], modelo, rep, th, titulo=f"ECG {ids[k]}",
                             reales=Yte[k], fichero=f"diagnostico_ecg_{ids[k]}.png")
    pd.DataFrame(resumen).to_csv(SALIDA / "resumen_modelos_finales.csv", index=False)
    dibujar_curvas_finales()
    print("Listo. Archivos en:", SALIDA.resolve())
