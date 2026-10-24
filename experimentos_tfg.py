"""Búsqueda experimental del TFG: learning rate, arquitectura, épocas y batch, desbalanceo y aumento de datos.

Cada entrenamiento se guarda en resultados.csv en cuanto termina. Si la ejecución se interrumpe,
al volver a lanzarla se saltan los entrenamientos ya hechos. La selección entre configuraciones
se hace siempre con el conjunto de validación; el conjunto de prueba solo se usa para informar.
"""
import os, json, time, random, hashlib, ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as sp_signal
from scipy.signal import sosfiltfilt, iirnotch, tf2sos
from sklearn.metrics import precision_recall_curve, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.regularizers import l2

DATA_DIR = Path(r"C:\Users\Usuario\Desktop\TFG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
OUT_BASE = Path("./outputs/experimentos")
OUT_DIR = OUT_BASE
DRY_RUN = False

FS = 100
SUPER_CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
NOMBRES = {"NORM": "Normal", "MI": "Infarto", "STTC": "Isquemia/ST-T", "HYP": "Hipertrofia", "CD": "Conducción"}
BANDPASS_LOW, BANDPASS_HIGH = 0.5, 40.0
NOTCH_FREQ, NOTCH_Q = 50.0, 30.0
STFT_NPERSEG, STFT_NOVERLAP = 64, 48
SPEC_SHAPE = (33, 64)
L2_REG = 1e-4

BASE = dict(
    representacion="espectrograma", arquitectura="A1_base_3bloques", lr=1e-3, batch=32,
    max_epocas=60, paciencia=8, perdida="pw_focal", gamma=1.0, sobremuestreo="ninguno",
    aumento="ninguno", semilla=42,
)


def configurar(data_dir=None, out_dir=None, dry_run=None):
    """Permite cambiar rutas y modo de prueba desde el notebook."""
    global DATA_DIR, OUT_BASE, OUT_DIR, DRY_RUN
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if out_dir is not None:
        OUT_BASE = Path(out_dir)
    if dry_run is not None:
        DRY_RUN = dry_run
    _DATOS.clear()
    OUT_DIR = OUT_BASE.parent / (OUT_BASE.name + "_prueba") if DRY_RUN else OUT_BASE
    (OUT_DIR / "historiales").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "modelos").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figuras").mkdir(parents=True, exist_ok=True)
    print("Datos:", DATA_DIR)
    print("Resultados:", OUT_DIR.resolve(), "| modo prueba" if DRY_RUN else "")


def cache_dir(nombre):
    d = DATA_DIR / "cache_v2" / nombre
    d.mkdir(parents=True, exist_ok=True)
    return d


def cargar_etiquetas():
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    df.columns = df.columns.str.strip()
    scp.columns = scp.columns.str.strip()
    scp_super = scp[scp["diagnostic_class"].notna()][["diagnostic_class"]]

    def superclases(s):
        if pd.isna(s):
            return []
        try:
            d = ast.literal_eval(s)
        except Exception:
            return []
        if not isinstance(d, dict):
            return []
        return list({scp_super.loc[k, "diagnostic_class"] for k in d if k in scp_super.index})

    df["superclasses"] = df["scp_codes"].apply(superclases)
    df = df[df["superclasses"].apply(len) > 0].reset_index()
    df["target_vec"] = df["superclasses"].apply(
        lambda c: np.array([1 if s in set(c) else 0 for s in SUPER_CLASSES], dtype=np.int32))
    train = df[df["strat_fold"].isin(range(1, 9))].reset_index(drop=True)
    val = df[df["strat_fold"] == 9].reset_index(drop=True)
    test = df[df["strat_fold"] == 10].reset_index(drop=True)
    assert not set(train.patient_id) & set(val.patient_id)
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(val.patient_id) & set(test.patient_id)
    if DRY_RUN:
        train, val, test = train.head(400), val.head(150), test.head(150)
    return train, val, test


def is_valid_signal(x, flat_std_thresh=1e-6):
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return False
    return np.std(x) >= flat_std_thresh


_BP_SOS = sp_signal.butter(4, [BANDPASS_LOW / (FS / 2), BANDPASS_HIGH / (FS / 2)], btype="band", output="sos")
_NOTCH_SOS = tf2sos(*iirnotch(NOTCH_FREQ / (FS / 2), NOTCH_Q))


def clean_lead(x):
    x = np.asarray(x, dtype=np.float64)
    if not is_valid_signal(x):
        return None
    x = sosfiltfilt(_BP_SOS, x)
    x = sosfiltfilt(_NOTCH_SOS, x)
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-8:
        return None
    return ((x - mu) / sigma).astype(np.float32)


def preprocess_ecg(raw):
    out = np.zeros_like(raw, dtype=np.float32)
    malas = 0
    for i in range(raw.shape[1]):
        c = clean_lead(raw[:, i])
        out[:, i] = c if c is not None else 0.0
        malas += c is None
    return None if malas > raw.shape[1] // 2 else out


def lead_to_spectrogram(x):
    _, _, Z = sp_signal.stft(x, fs=FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
    mag = np.log1p(np.abs(Z))
    F, T = SPEC_SHAPE
    mag = mag[:F, :T]
    if mag.shape != (F, T):
        pad = np.zeros((F, T), dtype=np.float32)
        pad[:mag.shape[0], :mag.shape[1]] = mag
        mag = pad
    mn, mx = mag.min(), mag.max()
    if mx - mn > 1e-8:
        mag = (mag - mn) / (mx - mn)
    return mag.astype(np.float32)


def ecg_to_tensor(sig):
    return np.stack([lead_to_spectrogram(sig[:, i]) for i in range(sig.shape[1])], axis=-1)


def _ruta(rel, carpeta):
    return carpeta / f"{hashlib.md5(rel.encode()).hexdigest()}.npy"


def senal_limpia(rel):
    cp = _ruta(rel, cache_dir("senal_limpia"))
    if cp.exists():
        return np.load(cp)
    raw, _ = wfdb.rdsamp(str(DATA_DIR / rel))
    clean = preprocess_ecg(raw)
    if clean is None:
        return None
    np.save(cp, clean)
    return clean


def espectrograma(rel):
    cp = _ruta(rel, cache_dir("con_filtrado"))
    if cp.exists():
        return np.load(cp)
    clean = senal_limpia(rel)
    if clean is None:
        return None
    t = ecg_to_tensor(clean)
    np.save(cp, t)
    return t


_DATOS = {}


def datos(representacion):
    """Devuelve (Xtr, Ytr, Xva, Yva, Xte, Yte, rutas_train) y los guarda en memoria entre entrenamientos."""
    if representacion in _DATOS:
        return _DATOS[representacion]
    for k in list(_DATOS):
        if k != "etiquetas":
            del _DATOS[k]
    if "etiquetas" not in _DATOS:
        _DATOS["etiquetas"] = cargar_etiquetas()
    tr, va, te = _DATOS["etiquetas"]
    f = espectrograma if representacion == "espectrograma" else senal_limpia

    def construir(d):
        X, Y, R = [], [], []
        for _, r in d.iterrows():
            t = f(r["filename_lr"])
            if t is None:
                continue
            X.append(t); Y.append(r["target_vec"]); R.append(r["filename_lr"])
        return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), R

    t0 = time.time()
    Xtr, Ytr, Rtr = construir(tr)
    Xva, Yva, _ = construir(va)
    Xte, Yte, _ = construir(te)
    print(f"Datos '{representacion}': {Xtr.shape}, {Xva.shape}, {Xte.shape} ({time.time() - t0:.0f} s)")
    _DATOS[representacion] = (Xtr, Ytr, Xva, Yva, Xte, Yte, Rtr)
    return _DATOS[representacion]


def aumentar_senal(sig, rng):
    """Copia sintética de un ECG limpio: estiramiento temporal, desplazamiento circular, deriva lenta y ruido."""
    n = sig.shape[0]
    factor = rng.uniform(0.9, 1.1)
    m = int(round(n * factor))
    est = sp_signal.resample(sig, m, axis=0)
    est = est[:n] if m >= n else np.pad(est, ((0, n - m), (0, 0)), mode="wrap")
    est = np.roll(est, rng.integers(0, n), axis=0)
    t = np.arange(n) / FS
    deriva = 0.1 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 2 * np.pi))
    est = est + deriva[:, None] + rng.normal(0, 0.05, est.shape)
    est = (est - est.mean(axis=0)) / (est.std(axis=0) + 1e-8)
    return est.astype(np.float32)


def copias_aumentadas_hyp(Rtr, Ytr, copias, semilla):
    """Genera 'copias' espectrogramas aumentados por cada ECG de entrenamiento con Hipertrofia."""
    carpeta = cache_dir(f"aumento_hyp_x{copias}_s{semilla}")
    idx = np.where(Ytr[:, SUPER_CLASSES.index("HYP")] == 1)[0]
    rng = np.random.default_rng(semilla)
    Xa, Ya = [], []
    for i in idx:
        for c in range(copias):
            cp = carpeta / f"{hashlib.md5((Rtr[i] + str(c)).encode()).hexdigest()}.npy"
            if cp.exists():
                t = np.load(cp)
            else:
                t = ecg_to_tensor(aumentar_senal(senal_limpia(Rtr[i]), rng))
                np.save(cp, t)
            Xa.append(t); Ya.append(Ytr[i])
    print(f"Aumento en señal: {len(Xa)} espectrogramas nuevos de Hipertrofia")
    return np.stack(Xa).astype(np.float32), np.stack(Ya).astype(np.float32)


class Lotes(tf.keras.utils.PyDataset):
    """Lotes de entrenamiento a partir de una lista de índices, con SpecAugment opcional."""

    def __init__(self, X, Y, indices, batch, spec_augment, semilla, **kw):
        super().__init__(**kw)
        self.X, self.Y, self.idx, self.batch = X, Y, np.asarray(indices), batch
        self.spec_augment = spec_augment
        self.rng = np.random.default_rng(semilla)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.batch))

    def on_epoch_end(self):
        self.orden = self.rng.permutation(self.idx)

    def __getitem__(self, i):
        sel = self.orden[i * self.batch:(i + 1) * self.batch]
        x = self.X[sel].copy()
        if self.spec_augment:
            F, T = x.shape[1], x.shape[2]
            for k in range(len(x)):
                if self.rng.random() < 0.5:
                    w = self.rng.integers(1, 9); t0 = self.rng.integers(0, T - w)
                    x[k, :, t0:t0 + w, :] = 0.0
                if self.rng.random() < 0.5:
                    w = self.rng.integers(1, 5); f0 = self.rng.integers(0, F - w)
                    x[k, f0:f0 + w, :, :] = 0.0
                x[k] = np.roll(x[k], self.rng.integers(-6, 7), axis=1)
        return x, self.Y[sel]


def _conv(x, f, k=3):
    x = layers.Conv2D(f, k, padding="same", activation="relu", kernel_regularizer=l2(L2_REG))(x)
    return layers.BatchNormalization()(x)


def _cabeza(x, n):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(n, activation="sigmoid")(x)


def _cnn(shape, n, filtros, k=3, doble=False):
    inp = layers.Input(shape=shape)
    x = inp
    drops = [0.2, 0.3, 0.3, 0.3]
    for j, f in enumerate(filtros):
        x = _conv(x, f, k)
        if doble:
            x = _conv(x, f, k)
        if j < len(filtros) - 1:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(drops[j])(x)
    return models.Model(inp, _cabeza(x, n))


def _residual(shape, n):
    inp = layers.Input(shape=shape)
    x = _conv(inp, 32)
    for j, f in enumerate([32, 64, 128]):
        atajo = layers.Conv2D(f, 1, padding="same", kernel_regularizer=l2(L2_REG))(x)
        y = _conv(x, f)
        y = layers.Conv2D(f, 3, padding="same", kernel_regularizer=l2(L2_REG))(y)
        y = layers.BatchNormalization()(y)
        x = layers.Activation("relu")(layers.Add()([y, atajo]))
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.2 + 0.1 * j)(x)
    return models.Model(inp, _cabeza(x, n))


def _crnn(shape, n):
    inp = layers.Input(shape=shape)
    x = layers.MaxPooling2D(2)(_conv(inp, 32))
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling2D(2)(_conv(x, 64))
    x = layers.Dropout(0.3)(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(0.3)(x)
    return models.Model(inp, layers.Dense(n, activation="sigmoid")(x))


def _cnn1d(shape, n):
    inp = layers.Input(shape=shape)
    x = inp
    for f, k, d in [(32, 7, 0.2), (64, 5, 0.3), (128, 5, 0.3)]:
        x = layers.Conv1D(f, k, padding="same", activation="relu", kernel_regularizer=l2(L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(d)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(0.3)(x)
    return models.Model(inp, layers.Dense(n, activation="sigmoid")(x))


ARQUITECTURAS = {
    "A1_base_3bloques": ("3 bloques 32-64-128 (modelo actual)", lambda s, n: _cnn(s, n, [32, 64, 128])),
    "A2_2bloques": ("2 bloques 32-64", lambda s, n: _cnn(s, n, [32, 64])),
    "A3_4bloques": ("4 bloques 32-64-128-256", lambda s, n: _cnn(s, n, [32, 64, 128, 256])),
    "A4_ancha": ("3 bloques anchos 64-128-256", lambda s, n: _cnn(s, n, [64, 128, 256])),
    "A5_doble_conv": ("3 bloques con 2 convoluciones (tipo VGG)", lambda s, n: _cnn(s, n, [32, 64, 128], doble=True)),
    "A6_residual": ("3 bloques residuales (tipo ResNet)", _residual),
    "A7_kernel_temporal": ("3 bloques con kernel 3×7 (más contexto temporal)", lambda s, n: _cnn(s, n, [32, 64, 128], k=(3, 7))),
    "A8_crnn": ("CNN + GRU bidireccional sobre el tiempo", _crnn),
    "A9_cnn1d_senal": ("CNN 1D sobre la señal temporal (sin STFT)", _cnn1d),
}


def perdida(pos_weight, gamma):
    pw = tf.constant(pos_weight, dtype=tf.float32)

    def f(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pos = pw * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        neg = (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(-(pos + neg))
    return f


def umbrales_optimos(Y, P):
    th = []
    for i in range(Y.shape[1]):
        prec, rec, thr = precision_recall_curve(Y[:, i], P[:, i])
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-8, None)
        th.append(float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5)
    return np.array(th)


def metricas(Y, P, th):
    pred = (P >= th).astype(int)
    out = {}
    f1s, precs, recs, aucs = [], [], [], []
    for i, c in enumerate(SUPER_CLASSES):
        tp = np.sum((Y[:, i] == 1) & (pred[:, i] == 1)); fp = np.sum((Y[:, i] == 0) & (pred[:, i] == 1))
        fn = np.sum((Y[:, i] == 1) & (pred[:, i] == 0))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        a = roc_auc_score(Y[:, i], P[:, i]) if len(np.unique(Y[:, i])) > 1 else np.nan
        out[f"f1_{c}"], out[f"auc_{c}"] = f, a
        f1s.append(f); precs.append(p); recs.append(r); aucs.append(a)
    out.update(f1=np.mean(f1s), precision=np.mean(precs), recall=np.mean(recs),
               auc=np.nanmean(aucs), mae=float(np.mean(np.abs(P - Y))))
    return out


class CurvaValidacion(callbacks.Callback):
    """Guarda en cada época el AUC macro y el F1 macro (umbral 0,5) de validación."""

    def __init__(self, Xva, Yva):
        super().__init__()
        self.Xva, self.Yva, self.filas = Xva, Yva, []

    def on_epoch_end(self, epoch, logs=None):
        P = self.model.predict(self.Xva, verbose=0, batch_size=256)
        m = metricas(self.Yva, P, np.full(P.shape[1], 0.5))
        self.filas.append(dict(epoca=epoch + 1, val_auc_macro=m["auc"], val_f1_05=m["f1"],
                               loss=float(logs.get("loss", np.nan)), val_loss=float(logs.get("val_loss", np.nan)),
                               lr=float(self.model.optimizer.learning_rate.numpy())))


CLAVES = list(BASE.keys())


def clave(cfg):
    return hashlib.md5(json.dumps({k: cfg[k] for k in CLAVES}, sort_keys=True).encode()).hexdigest()[:12]


def resultados():
    f = OUT_DIR / "resultados.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def entrenar(id_, etapa, descripcion, **cambios):
    """Entrena una configuración (BASE + cambios), evalúa y guarda una fila en resultados.csv."""
    cfg = {**BASE, **cambios}
    k = clave(cfg)
    res = resultados()
    if len(res) and id_ in set(res["id"]):
        print(f"[ya hecho] {id_}")
        return res[res["id"] == id_].iloc[0].to_dict()
    if len(res) and k in set(res["clave"]):
        fila = res[res["clave"] == k].iloc[0].to_dict()
        fila.update(id=id_, etapa=etapa, descripcion=descripcion, reutilizado_de=fila["id"])
        pd.concat([res, pd.DataFrame([fila])], ignore_index=True).to_csv(OUT_DIR / "resultados.csv", index=False)
        print(f"[reutilizado] {id_} = {fila['reutilizado_de']} (misma configuración)")
        return fila

    print(f"\n=== {id_}: {descripcion} ===")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg["semilla"])
    rep = "espectrograma" if cfg["arquitectura"] != "A9_cnn1d_senal" else "senal"
    Xtr, Ytr, Xva, Yva, Xte, Yte, Rtr = datos(rep)

    if cfg["aumento"] in ("senal_hyp_x2", "senal_hyp_x2+specaugment"):
        Xa, Ya = copias_aumentadas_hyp(Rtr, Ytr, 2, cfg["semilla"])
        Xtr_, Ytr_ = np.concatenate([Xtr, Xa]), np.concatenate([Ytr, Ya])
    else:
        Xtr_, Ytr_ = Xtr, Ytr
    indices = np.arange(len(Xtr_))
    if cfg["sobremuestreo"] == "hyp_x3":
        hyp = np.where(Ytr_[:, SUPER_CLASSES.index("HYP")] == 1)[0]
        indices = np.concatenate([indices, hyp, hyp])
    Yefectivo = Ytr_[indices]

    if cfg["perdida"] in ("pw_focal", "pw_bce"):
        pos = Yefectivo.sum(0); neg = len(Yefectivo) - pos
        pw = np.clip(neg / np.maximum(pos, 1), 1.0, 10.0)
    else:
        pw = np.ones(len(SUPER_CLASSES))
    gamma = cfg["gamma"] if cfg["perdida"] in ("pw_focal", "focal") else 0.0

    model = ARQUITECTURAS[cfg["arquitectura"]][1](Xtr.shape[1:], len(SUPER_CLASSES))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss=perdida(pw, gamma),
                  metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)])
    curva = CurvaValidacion(Xva, Yva)
    cbs = [curva, callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-6)]
    if cfg["paciencia"]:
        cbs.append(callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=int(cfg["paciencia"]),
                                           restore_best_weights=True))
    spec_aug = "specaugment" in cfg["aumento"]
    if len(indices) == len(Xtr_) and not spec_aug:
        entrada = dict(x=Xtr_, y=Ytr_, batch_size=int(cfg["batch"]), shuffle=True)
    else:
        entrada = dict(x=Lotes(Xtr_, Ytr_, indices, int(cfg["batch"]), spec_aug, cfg["semilla"]))
    max_ep = 2 if DRY_RUN else int(cfg["max_epocas"])
    t0 = time.time()
    model.fit(**entrada, validation_data=(Xva, Yva), epochs=max_ep, callbacks=cbs, verbose=2)
    minutos = (time.time() - t0) / 60

    Pva = model.predict(Xva, verbose=0, batch_size=256)
    Pte = model.predict(Xte, verbose=0, batch_size=256)
    th = umbrales_optimos(Yva, Pva)
    mva05 = metricas(Yva, Pva, np.full(5, 0.5))
    mva = metricas(Yva, Pva, th)
    mte = metricas(Yte, Pte, th)
    curva_df = pd.DataFrame(curva.filas)
    mejor_ep = int(curva_df["val_auc_macro"].idxmax()) + 1 if cfg["paciencia"] else len(curva_df)

    fila = dict(id=id_, etapa=etapa, descripcion=descripcion, clave=k, reutilizado_de="", **cfg,
                parametros=int(model.count_params()), n_train=int(len(indices)),
                epocas_entrenadas=len(curva_df), mejor_epoca=mejor_ep, minutos=round(minutos, 1),
                val_auc=mva["auc"], val_f1_05=mva05["f1"], val_f1_opt=mva["f1"],
                umbrales=json.dumps([round(float(t), 3) for t in th]),
                test_auc=mte["auc"], test_f1=mte["f1"], test_precision=mte["precision"],
                test_recall=mte["recall"], test_mae=mte["mae"],
                **{f"test_f1_{c}": mte[f"f1_{c}"] for c in SUPER_CLASSES},
                **{f"test_auc_{c}": mte[f"auc_{c}"] for c in SUPER_CLASSES},
                **{f"val_f1_{c}": mva[f"f1_{c}"] for c in SUPER_CLASSES})
    curva_df.to_csv(OUT_DIR / "historiales" / f"{id_}.csv", index=False)
    model.save_weights(OUT_DIR / "modelos" / f"{id_}.weights.h5")
    res = resultados()
    pd.concat([res, pd.DataFrame([fila])], ignore_index=True).to_csv(OUT_DIR / "resultados.csv", index=False)
    print(f"--> {id_}: val AUC {mva['auc']:.4f} | val F1 {mva['f1']:.4f} | {minutos:.1f} min, mejor época {mejor_ep}")
    return fila


def mejor(etapa, excluir=()):
    res = resultados()
    r = res[(res["etapa"] == etapa) & (~res["id"].isin(excluir))]
    return r.sort_values(["val_auc", "val_f1_opt"], ascending=False).iloc[0].to_dict()


def cfg_de(fila):
    out = {k: fila[k] for k in CLAVES}
    for k in ("lr", "gamma"):
        out[k] = float(out[k])
    for k in ("batch", "max_epocas", "semilla"):
        out[k] = int(out[k])
    out["paciencia"] = None if pd.isna(out["paciencia"]) or out["paciencia"] in ("", "None") else int(out["paciencia"])
    return out


def etapa1_learning_rate():
    for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        entrenar(f"S1_lr_{lr:g}", "1_learning_rate", f"Learning rate {lr:g}", lr=lr)
    b = mejor("1_learning_rate")
    print("Mejor learning rate (validación):", b["lr"])
    return b


def etapa2_arquitectura():
    base = cfg_de(mejor("1_learning_rate"))
    for nombre, (desc, _) in ARQUITECTURAS.items():
        entrenar(f"S2_{nombre}", "2_arquitectura", desc, **{**base, "arquitectura": nombre})
    b = mejor("2_arquitectura", excluir=["S2_A9_cnn1d_senal"])
    print("Mejor arquitectura sobre el espectrograma (validación):", b["arquitectura"])
    return b


def etapa3_batch_epocas():
    base = cfg_de(mejor("2_arquitectura", excluir=["S2_A9_cnn1d_senal"]))
    entrenar("S3_referencia", "3_batch_epocas", f"Referencia (batch {base['batch']}, paciencia {base['paciencia']})", **base)
    for b in [16, 64]:
        entrenar(f"S3_batch_{b}", "3_batch_epocas", f"Batch {b}", **{**base, "batch": b})
    for p in [4, 15]:
        entrenar(f"S3_paciencia_{p}", "3_batch_epocas", f"Parada temprana con paciencia {p}",
                 **{**base, "paciencia": p, "max_epocas": 100})
    entrenar("S3_sin_parada_80", "3_curva_epocas", "80 épocas sin parada temprana", **{**base, "paciencia": None, "max_epocas": 80})
    b = mejor("3_batch_epocas")
    print("Mejor batch/paciencia (validación):", b["batch"], b["paciencia"])
    return b


def etapa4_desbalanceo():
    base = cfg_de(mejor("3_batch_epocas"))
    variantes = [
        ("S4_sin_balanceo", "Sin tratamiento (entropía cruzada)", dict(perdida="bce", gamma=0.0)),
        ("S4_pos_weight", "Solo pos_weight", dict(perdida="pw_bce", gamma=0.0)),
        ("S4_focal_g2", "Solo focal loss (γ = 2)", dict(perdida="focal", gamma=2.0)),
        ("S4_pw_focal_g1", "pos_weight + focal loss (γ = 1)", dict(perdida="pw_focal", gamma=1.0)),
        ("S4_pw_focal_g2", "pos_weight + focal loss (γ = 2)", dict(perdida="pw_focal", gamma=2.0)),
        ("S4_sobremuestreo_hyp", "Sobremuestreo de Hipertrofia ×3, sin ponderar", dict(perdida="bce", gamma=0.0, sobremuestreo="hyp_x3")),
        ("S4_specaugment", "SpecAugment + pos_weight + focal", dict(perdida="pw_focal", gamma=1.0, aumento="specaugment")),
        ("S4_aumento_senal_hyp", "Aumento en señal de Hipertrofia ×2 + pos_weight + focal", dict(perdida="pw_focal", gamma=1.0, aumento="senal_hyp_x2")),
        ("S4_aumento_senal_y_spec", "Aumento en señal + SpecAugment + pos_weight + focal", dict(perdida="pw_focal", gamma=1.0, aumento="senal_hyp_x2+specaugment")),
    ]
    for id_, desc, cambios in variantes:
        entrenar(id_, "4_desbalanceo", desc, **{**base, **cambios})
    b = mejor("4_desbalanceo")
    print("Mejor técnica de desbalanceo (validación):", b["descripcion"])
    return b


def etapa5_semillas(semillas=(42, 7, 123)):
    final = cfg_de(mejor("4_desbalanceo"))
    referencia = dict(BASE)
    for s in semillas:
        entrenar(f"S5_final_s{s}", "5_semillas_final", "Configuración final", **{**final, "semilla": s})
        entrenar(f"S5_referencia_s{s}", "5_semillas_referencia", "Modelo 05 (referencia)", **{**referencia, "semilla": s})


def ejecutar_todo():
    etapa1_learning_rate()
    etapa2_arquitectura()
    etapa3_batch_epocas()
    etapa4_desbalanceo()
    etapa5_semillas()
    tablas_y_figuras()


def _fmt(df):
    return df.round(3)


def tablas_y_figuras():
    """Crea tablas (Excel y CSV) y figuras para la memoria a partir de resultados.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = resultados()
    if not len(res):
        print("Todavía no hay resultados.")
        return
    cols = ["descripcion", "parametros", "mejor_epoca", "minutos", "val_auc", "val_f1_05", "val_f1_opt"]
    nombres = {"descripcion": "Configuración", "parametros": "Parámetros", "mejor_epoca": "Mejor época",
               "minutos": "Minutos", "val_auc": "AUC val.", "val_f1_05": "F1 val. (0,5)",
               "val_f1_opt": "F1 val. (umbral óptimo)", "test_auc": "AUC test", "test_f1": "F1 test",
               "test_f1_HYP": "F1 HYP test", "test_recall": "Recall test", "test_precision": "Precisión test"}
    tablas = {}
    for etapa, hoja in [("1_learning_rate", "1 Learning rate"), ("2_arquitectura", "2 Arquitectura"),
                        ("3_batch_epocas", "3 Batch y paciencia"), ("4_desbalanceo", "4 Desbalanceo")]:
        r = res[res["etapa"] == etapa]
        if len(r):
            t = _fmt(r[cols + ["val_f1_HYP"]].rename(columns={**nombres, "val_f1_HYP": "F1 HYP val."}))
            tablas[hoja] = t.sort_values("AUC val.", ascending=False)

    curva_f = OUT_DIR / "historiales" / "S3_sin_parada_80.csv"
    if curva_f.exists():
        c = pd.read_csv(curva_f)
        sel = c[c["epoca"].isin([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80])]
        tablas["3b Épocas"] = _fmt(sel[["epoca", "val_auc_macro", "val_f1_05", "loss", "val_loss", "lr"]].rename(
            columns={"epoca": "Época", "val_auc_macro": "AUC val.", "val_f1_05": "F1 val. (0,5)",
                     "loss": "Pérdida entren.", "val_loss": "Pérdida val.", "lr": "Learning rate"}))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(c["epoca"], c["loss"], label="Entrenamiento", color="#8a96a3")
        ax[0].plot(c["epoca"], c["val_loss"], label="Validación", color="#1f5f7a")
        ax[0].set_xlabel("Época"); ax[0].set_ylabel("Pérdida"); ax[0].legend(frameon=False)
        ax[1].plot(c["epoca"], c["val_auc_macro"], color="#1f5f7a", label="AUC val.")
        ax[1].plot(c["epoca"], c["val_f1_05"], color="#c42d3a", label="F1 val. (0,5)")
        ax[1].set_xlabel("Época"); ax[1].legend(frameon=False)
        for a in ax:
            a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "figuras" / "curva_epocas.png", dpi=200); plt.close(fig)

    sem = res[res["etapa"].str.startswith("5_semillas")]
    if len(sem):
        g = sem.groupby("etapa")[["test_auc", "test_f1", "test_precision", "test_recall", "test_f1_HYP"]].agg(["mean", "std"])
        nom = {"test_auc": "AUC test", "test_f1": "F1 macro test", "test_precision": "Precisión test",
               "test_recall": "Recall test", "test_f1_HYP": "F1 HYP test"}
        g.columns = [f"{nom[a]} ({'media' if b == 'mean' else 'desv. típica'})" for a, b in g.columns]
        g.index.name = "Modelo"
        tablas["5 Final vs referencia"] = _fmt(g.reset_index().replace(
            {"5_semillas_final": "Configuración final (3 semillas)", "5_semillas_referencia": "Modelo 05 (3 semillas)"}))

    todo = res[["id", "etapa", "descripcion", "parametros", "val_auc", "val_f1_opt", "test_auc", "test_f1",
                "test_precision", "test_recall", "test_mae"] + [f"test_f1_{c}" for c in SUPER_CLASSES]]
    tablas["Todos (incl. test)"] = _fmt(todo)

    with pd.ExcelWriter(OUT_DIR / "tablas_memoria.xlsx") as w:
        for hoja, t in tablas.items():
            t.to_excel(w, sheet_name=hoja[:31], index=False)

    for etapa, titulo in [("1_learning_rate", "Learning rate"), ("2_arquitectura", "Arquitectura"),
                          ("3_batch_epocas", "Batch y paciencia"), ("4_desbalanceo", "Desbalanceo y aumento")]:
        r = res[res["etapa"] == etapa].sort_values("val_auc")
        if not len(r):
            continue
        fig, ax = plt.subplots(1, 2, figsize=(11, 0.45 * len(r) + 1.4), sharey=True)
        y = np.arange(len(r))
        b = r["val_auc"].idxmax()
        col = ["#c42d3a" if i == b else "#5d7c93" for i in r.index]
        ax[0].barh(y, r["val_auc"], color=col); ax[0].set_title("AUC macro (validación)", loc="left", fontsize=10)
        ax[1].barh(y, r["val_f1_opt"], color=col); ax[1].set_title("F1 macro (validación, umbral óptimo)", loc="left", fontsize=10)
        ax[0].set_yticks(y); ax[0].set_yticklabels(r["descripcion"], fontsize=8)
        ax[0].set_xlim(max(0.5, r["val_auc"].min() - 0.03), min(1, r["val_auc"].max() + 0.01))
        ax[1].set_xlim(max(0, r["val_f1_opt"].min() - 0.05), min(1, r["val_f1_opt"].max() + 0.02))
        for a, cname in zip(ax, ["val_auc", "val_f1_opt"]):
            for yy, v in zip(y, r[cname]):
                a.text(v, yy, f" {v:.3f}".replace(".", ","), va="center", fontsize=8)
            a.spines[["top", "right"]].set_visible(False)
        fig.suptitle(titulo, x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT_DIR / "figuras" / f"etapa_{etapa}.png", dpi=200); plt.close(fig)

    print("Tablas en", (OUT_DIR / "tablas_memoria.xlsx").resolve())
    print("Figuras en", (OUT_DIR / "figuras").resolve())
    return tablas
