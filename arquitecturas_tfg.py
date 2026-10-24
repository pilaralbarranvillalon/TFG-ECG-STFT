"""Segunda búsqueda del TFG: comparación amplia de arquitecturas y control del sobreajuste.

Reutiliza los datos, la caché, las métricas y los umbrales de experimentos_tfg.py (deben estar en la
misma carpeta). Cada entrenamiento guarda su fila en resultados.csv y su curva por época en
historiales/, de modo que si la ejecución se interrumpe, al volver a lanzarla se saltan los ya hechos.

Etapa 6: 15 arquitecturas nuevas (9 sobre el espectrograma y 6 sobre la señal temporal), con el mismo
         protocolo para todas.
Etapa 7: técnicas contra el sobreajuste sobre la mejor arquitectura de cada representación.
Etapa 8: la configuración final de cada representación con tres semillas.

La elección se hace siempre con el conjunto de validación; el conjunto de prueba solo se usa para informar.
"""
import json, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.regularizers import l2

import experimentos_tfg as X

SUPER_CLASSES = X.SUPER_CLASSES
OUT_DIR = Path("./outputs/arquitecturas")
R = {"l2": 1e-4, "drop": 1.0}

BASE = dict(
    arquitectura="E01_base_3bloques", lr=3e-4, batch=32, max_epocas=80, paciencia=15,
    monitor="val_auc", perdida="pw_focal", gamma=1.0, aumento="ninguno", drop=1.0, l2=1e-4,
    suavizado=0.0, optimizador="adam", weight_decay=0.0, semilla=42,
)
CLAVES = list(BASE.keys())


def configurar(data_dir, dry_run=False):
    global OUT_DIR
    X.configurar(data_dir=data_dir, out_dir="./outputs/arquitecturas", dry_run=dry_run)
    OUT_DIR = X.OUT_DIR
    print("Resultados de esta búsqueda en:", OUT_DIR.resolve())


def _d(p):
    return min(0.7, p * R["drop"])


def _reg():
    return l2(R["l2"]) if R["l2"] else None


def _cbr(x, f, k=3, sep=False):
    if sep:
        x = layers.SeparableConv2D(f, k, padding="same", use_bias=False,
                                   depthwise_regularizer=_reg(), pointwise_regularizer=_reg())(x)
    else:
        x = layers.Conv2D(f, k, padding="same", use_bias=False, kernel_regularizer=_reg())(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def _cbr1(x, f, k, d=1):
    x = layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False, kernel_regularizer=_reg())(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def _cabeza(x, n):
    x = layers.Dropout(_d(0.4))(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=_reg())(x)
    x = layers.Dropout(_d(0.3))(x)
    return layers.Dense(n, activation="sigmoid")(x)


def _se(x, r=8):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x) if len(x.shape) == 4 else layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(c // r, 4), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, c) if len(x.shape) == 4 else (1, c))(s)
    return layers.Multiply()([x, s])


class PosicionAprendida(layers.Layer):
    """Suma a cada instante de la secuencia un vector de posición aprendido."""

    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, input_shape[1], input_shape[2]),
                                   initializer="random_normal")

    def call(self, x):
        return x + self.pos


def _transformer(x, capas=2, dim=64, cabezas=4):
    x = layers.Dense(dim)(x)
    x = PosicionAprendida()(x)
    for _ in range(capas):
        a = layers.MultiHeadAttention(num_heads=cabezas, key_dim=dim // cabezas, dropout=_d(0.1))(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, layers.Dropout(_d(0.1))(a)]))
        f = layers.Dense(2 * dim, activation="relu")(x)
        f = layers.Dense(dim)(f)
        x = layers.LayerNormalization()(layers.Add()([x, layers.Dropout(_d(0.1))(f)]))
    return x


def _a_secuencia(x):
    x = layers.Permute((2, 1, 3))(x)
    return layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)


def e01_base(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for j, f in enumerate([32, 64, 128]):
        x = _cbr(x, f)
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(_d(0.2 + 0.1 * (j > 0)))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def e02_mobilenet(s, n):
    inp = layers.Input(shape=s)
    x = _cbr(inp, 32)
    for j, f in enumerate([64, 128, 256]):
        x = _cbr(x, f, sep=True)
        x = _cbr(x, f, sep=True)
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(_d(0.2 + 0.1 * (j > 0)))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def e03_se(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for j, f in enumerate([32, 64, 128]):
        x = _se(_cbr(x, f))
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(_d(0.2 + 0.1 * (j > 0)))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def _modulo_inception(x, f):
    a = _cbr(x, f, 1)
    b = _cbr(_cbr(x, f, 1), f, 3)
    c = _cbr(_cbr(_cbr(x, f, 1), f, 3), f, 3)
    d = _cbr(layers.MaxPooling2D(3, strides=1, padding="same")(x), f, 1)
    return layers.Concatenate()([a, b, c, d])


def e04_inception(s, n):
    inp = layers.Input(shape=s)
    x = _cbr(inp, 32)
    for j, f in enumerate([16, 32, 32]):
        x = _modulo_inception(x, f)
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(_d(0.2 + 0.1 * (j > 0)))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def e05_densenet(s, n):
    inp = layers.Input(shape=s)
    x = _cbr(inp, 32)
    for j in range(3):
        for _ in range(4):
            x = layers.Concatenate()([x, _cbr(x, 16)])
        if j < 2:
            x = _cbr(x, x.shape[-1] // 2, 1)
            x = layers.AveragePooling2D(2)(x)
            x = layers.Dropout(_d(0.2))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def _mbconv(x, f, expansion=4):
    entrada = x
    c = x.shape[-1]
    y = _cbr(x, c * expansion, 1)
    y = layers.DepthwiseConv2D(3, padding="same", use_bias=False, depthwise_regularizer=_reg())(y)
    y = layers.Activation("relu")(layers.BatchNormalization()(y))
    y = _se(y, r=16)
    y = layers.Conv2D(f, 1, use_bias=False, kernel_regularizer=_reg())(y)
    y = layers.BatchNormalization()(y)
    return layers.Add()([entrada, y]) if c == f else y


def e06_efficientnet(s, n):
    inp = layers.Input(shape=s)
    x = _cbr(inp, 24)
    for j, f in enumerate([24, 48, 96]):
        x = _mbconv(x, f)
        x = _mbconv(x, f)
        if j < 2:
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(_d(0.2))(x)
    x = _cbr(x, 192, 1)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling2D()(x), n))


def e07_cnn_transformer(s, n):
    inp = layers.Input(shape=s)
    x = layers.MaxPooling2D(2)(_cbr(inp, 32))
    x = layers.Dropout(_d(0.2))(x)
    x = layers.MaxPooling2D((2, 1))(_cbr(x, 64))
    x = layers.Dropout(_d(0.3))(x)
    x = _transformer(_a_secuencia(x))
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


def e08_cnn_bilstm(s, n):
    inp = layers.Input(shape=s)
    x = layers.MaxPooling2D(2)(_cbr(inp, 32))
    x = layers.Dropout(_d(0.2))(x)
    x = layers.MaxPooling2D((2, 1))(_cbr(x, 64))
    x = layers.Dropout(_d(0.3))(x)
    x = layers.Bidirectional(layers.LSTM(64))(_a_secuencia(x))
    return models.Model(inp, _cabeza(x, n))


def e09_por_derivacion(s, n):
    inp = layers.Input(shape=s)
    x = layers.Permute((3, 1, 2))(inp)
    x = layers.Reshape((s[2], s[0], s[1], 1))(x)
    codificador = models.Sequential([
        layers.Input(shape=(s[0], s[1], 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=_reg()),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=_reg()),
        layers.BatchNormalization(), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=_reg()),
        layers.BatchNormalization(), layers.GlobalAveragePooling2D(),
    ], name="codificador_derivacion")
    x = layers.TimeDistributed(codificador)(x)
    x = layers.Flatten()(x)
    return models.Model(inp, _cabeza(x, n))


def s01_cnn1d(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for f, k, d in [(32, 7, 0.2), (64, 5, 0.3), (128, 5, 0.3)]:
        x = layers.MaxPooling1D(2)(_cbr1(x, f, k))
        x = layers.Dropout(_d(d))(x)
    x = _cbr1(x, 128, 3)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


def s02_resnet1d(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for j, f in enumerate([64, 128, 128]):
        atajo = layers.BatchNormalization()(layers.Conv1D(f, 1, padding="same", kernel_regularizer=_reg())(x))
        y = _cbr1(x, f, 8)
        y = _cbr1(y, f, 5)
        y = layers.Conv1D(f, 3, padding="same", use_bias=False, kernel_regularizer=_reg())(y)
        y = layers.BatchNormalization()(y)
        x = layers.Activation("relu")(layers.Add()([y, atajo]))
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(_d(0.2))(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


def _modulo_inception1d(x, f=32):
    b = layers.Conv1D(f, 1, padding="same", use_bias=False)(x) if x.shape[-1] > 1 else x
    ramas = [layers.Conv1D(f, k, padding="same", use_bias=False, kernel_regularizer=_reg())(b) for k in (9, 19, 39)]
    ramas.append(layers.Conv1D(f, 1, padding="same", use_bias=False)(layers.MaxPooling1D(3, strides=1, padding="same")(x)))
    y = layers.Concatenate()(ramas)
    return layers.Activation("relu")(layers.BatchNormalization()(y))


def s03_inceptiontime(s, n):
    inp = layers.Input(shape=s)
    x = inp
    residuo = inp
    for j in range(6):
        x = _modulo_inception1d(x)
        if j % 3 == 2:
            atajo = layers.BatchNormalization()(layers.Conv1D(x.shape[-1], 1, padding="same", use_bias=False)(residuo))
            x = layers.Activation("relu")(layers.Add()([x, atajo]))
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(_d(0.2))(x)
            residuo = x
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


def s04_cnn1d_bilstm(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for f, k in [(32, 7), (64, 5), (128, 5)]:
        x = layers.MaxPooling1D(2)(_cbr1(x, f, k))
        x = layers.Dropout(_d(0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    return models.Model(inp, _cabeza(x, n))


def s05_transformer1d(s, n):
    inp = layers.Input(shape=s)
    x = inp
    for f, k in [(32, 7), (64, 5), (64, 5)]:
        x = layers.MaxPooling1D(2)(_cbr1(x, f, k))
    x = layers.Dropout(_d(0.2))(x)
    x = _transformer(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


def s06_tcn(s, n):
    inp = layers.Input(shape=s)
    x = _cbr1(inp, 64, 7)
    for d in (1, 2, 4, 8, 16):
        y = _cbr1(x, 64, 5, d)
        y = layers.Dropout(_d(0.1))(y)
        y = _cbr1(y, 64, 5, d)
        x = layers.Add()([x, y])
        if d in (2, 8):
            x = layers.MaxPooling1D(2)(x)
    return models.Model(inp, _cabeza(layers.GlobalAveragePooling1D()(x), n))


ARQUITECTURAS = {
    "E01_base_3bloques": ("espectrograma", "CNN 2D de 3 bloques 32-64-128 (referencia)", e01_base),
    "E02_mobilenet": ("espectrograma", "Convoluciones separables (tipo MobileNet)", e02_mobilenet),
    "E03_se": ("espectrograma", "CNN 2D con atención por canal (SE)", e03_se),
    "E04_inception": ("espectrograma", "Módulos multiescala (tipo Inception)", e04_inception),
    "E05_densenet": ("espectrograma", "Bloques densos (tipo DenseNet)", e05_densenet),
    "E06_efficientnet": ("espectrograma", "Bloques MBConv con SE (tipo EfficientNet)", e06_efficientnet),
    "E07_cnn_transformer": ("espectrograma", "CNN 2D + Transformer sobre el tiempo", e07_cnn_transformer),
    "E08_cnn_bilstm": ("espectrograma", "CNN 2D + LSTM bidireccional", e08_cnn_bilstm),
    "E09_por_derivacion": ("espectrograma", "Codificador compartido por derivación", e09_por_derivacion),
    "S01_cnn1d": ("senal", "CNN 1D de 4 bloques (referencia)", s01_cnn1d),
    "S02_resnet1d": ("senal", "ResNet 1D (tipo Wang)", s02_resnet1d),
    "S03_inceptiontime": ("senal", "InceptionTime", s03_inceptiontime),
    "S04_cnn1d_bilstm": ("senal", "CNN 1D + LSTM bidireccional", s04_cnn1d_bilstm),
    "S05_transformer1d": ("senal", "CNN 1D + Transformer", s05_transformer1d),
    "S06_tcn": ("senal", "Red convolucional temporal dilatada (TCN)", s06_tcn),
}


def perdida(pos_weight, gamma, suavizado):
    pw = tf.constant(pos_weight, dtype=tf.float32)

    def f(y_true, y_pred):
        y = y_true * (1.0 - suavizado) + 0.5 * suavizado
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pos = pw * y * tf.pow(1 - p, gamma) * tf.math.log(p)
        neg = (1 - y) * tf.pow(p, gamma) * tf.math.log(1 - p)
        return tf.reduce_mean(-(pos + neg))
    return f


class LotesSenal(X.Lotes):
    """Lotes de señal temporal con aumento opcional: escala por derivación, ruido y desplazamiento circular."""

    def __getitem__(self, i):
        sel = self.orden[i * self.batch:(i + 1) * self.batch]
        x = self.X[sel].copy()
        if self.spec_augment:
            n, T, C = x.shape
            x *= self.rng.uniform(0.9, 1.1, size=(n, 1, C)).astype(np.float32)
            x += self.rng.normal(0, 0.03, size=x.shape).astype(np.float32)
            for k in range(n):
                x[k] = np.roll(x[k], self.rng.integers(-50, 51), axis=0)
        return x, self.Y[sel]


class Curva(callbacks.Callback):
    """Guarda en cada época las pérdidas de Keras, el AUC y el F1 de validación y las pérdidas sin dropout.

    perdida_entren_eval y perdida_val_eval se calculan con el modelo en modo evaluación (sin dropout ni
    aumento de datos) y con la misma función de pérdida, sobre una muestra fija del entrenamiento del mismo
    tamaño que la validación; su diferencia mide el sobreajuste de forma comparable entre configuraciones.
    """

    def __init__(self, Xva, Yva, Xtr, Ytr, fn):
        super().__init__()
        idx = np.random.default_rng(0).choice(len(Xtr), size=min(len(Xtr), len(Xva)), replace=False)
        self.Xva, self.Yva, self.Xtr, self.Ytr, self.fn = Xva, Yva, Xtr[idx], Ytr[idx], fn
        self.filas = []

    def on_epoch_end(self, epoch, logs=None):
        Pva = self.model.predict(self.Xva, verbose=0, batch_size=256)
        Ptr = self.model.predict(self.Xtr, verbose=0, batch_size=256)
        m = X.metricas(self.Yva, Pva, np.full(Pva.shape[1], 0.5))
        self.filas.append(dict(
            epoca=epoch + 1, val_auc_macro=m["auc"], val_f1_05=m["f1"],
            loss=float(logs.get("loss", np.nan)), val_loss=float(logs.get("val_loss", np.nan)),
            perdida_entren_eval=float(self.fn(tf.constant(self.Ytr), tf.constant(Ptr))),
            perdida_val_eval=float(self.fn(tf.constant(self.Yva), tf.constant(Pva))),
            lr=float(self.model.optimizer.learning_rate.numpy())))


def clave(cfg):
    return hashlib.md5(json.dumps({k: cfg[k] for k in CLAVES}, sort_keys=True).encode()).hexdigest()[:12]


def resultados():
    f = OUT_DIR / "resultados.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def sobreajuste(curva, mejor_ep):
    """Indicadores de sobreajuste a partir de la curva por época."""
    c = curva.set_index("epoca")
    vl = c["val_loss"]
    return dict(
        brecha=float(c.loc[mejor_ep, "perdida_val_eval"] - c.loc[mejor_ep, "perdida_entren_eval"]),
        brecha_keras=float(c.loc[mejor_ep, "val_loss"] - c.loc[mejor_ep, "loss"]),
        epoca_min_val_loss=int(vl.idxmin()),
        val_loss_min=float(vl.min()),
        val_loss_mejor=float(c.loc[mejor_ep, "val_loss"]),
        exceso_val_loss=float(c.loc[mejor_ep, "val_loss"] - vl.min()),
        subida_final=float(vl.iloc[-1] - vl.min()),
    )


def entrenar(id_, etapa, descripcion, **cambios):
    """Entrena BASE + cambios, fija los umbrales en validación, evalúa y guarda una fila en resultados.csv."""
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
    R["l2"], R["drop"] = cfg["l2"], cfg["drop"]
    rep, _, constructor = ARQUITECTURAS[cfg["arquitectura"]]
    Xtr, Ytr, Xva, Yva, Xte, Yte, _ = X.datos(rep)

    if cfg["perdida"] == "pw_focal":
        pos = Ytr.sum(axis=0)
        pw = np.clip((len(Ytr) - pos) / np.maximum(pos, 1), 1.0, 10.0)
        gamma = cfg["gamma"]
    else:
        pw, gamma = np.ones(len(SUPER_CLASSES)), 0.0

    model = constructor(Xtr.shape[1:], len(SUPER_CLASSES))
    if cfg["optimizador"] == "adamw":
        opt = tf.keras.optimizers.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = tf.keras.optimizers.Adam(cfg["lr"])
    fn = perdida(pw, gamma, cfg["suavizado"])
    model.compile(optimizer=opt, loss=fn,
                  metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)])

    modo = "max" if cfg["monitor"] == "val_auc" else "min"
    curva = Curva(Xva, Yva, Xtr, Ytr, fn)
    cbs = [curva,
           callbacks.ReduceLROnPlateau(monitor=cfg["monitor"], mode=modo, factor=0.5, patience=5, min_lr=1e-6),
           callbacks.EarlyStopping(monitor=cfg["monitor"], mode=modo, patience=int(cfg["paciencia"]),
                                   restore_best_weights=True)]
    aumentar = cfg["aumento"] != "ninguno"
    clase = X.Lotes if rep == "espectrograma" else LotesSenal
    lotes = clase(Xtr, Ytr, np.arange(len(Xtr)), int(cfg["batch"]), aumentar, cfg["semilla"])
    max_ep = 2 if X.DRY_RUN else int(cfg["max_epocas"])
    t0 = time.time()
    hist = model.fit(lotes, validation_data=(Xva, Yva), epochs=max_ep, callbacks=cbs, verbose=2)
    minutos = (time.time() - t0) / 60

    curva_df = pd.DataFrame(curva.filas)
    serie = np.array(hist.history[cfg["monitor"]])
    mejor_ep = int(np.argmax(serie) if modo == "max" else np.argmin(serie)) + 1
    if len(curva_df) > len(serie):
        curva_df = curva_df.iloc[:len(serie)]

    Pva = model.predict(Xva, verbose=0, batch_size=256)
    Pte = model.predict(Xte, verbose=0, batch_size=256)
    th = X.umbrales_optimos(Yva, Pva)
    mva = X.metricas(Yva, Pva, th)
    mte = X.metricas(Yte, Pte, th)

    fila = dict(id=id_, etapa=etapa, descripcion=descripcion, clave=k, reutilizado_de="", representacion=rep,
                **cfg, parametros=int(model.count_params()), epocas_entrenadas=len(curva_df),
                mejor_epoca=mejor_ep, minutos=round(minutos, 1),
                val_auc=mva["auc"], val_f1_opt=mva["f1"],
                umbrales=json.dumps([round(float(t), 3) for t in th]),
                test_auc=mte["auc"], test_f1=mte["f1"], test_precision=mte["precision"],
                test_recall=mte["recall"], test_mae=mte["mae"],
                **sobreajuste(curva_df, mejor_ep),
                **{f"test_f1_{c}": mte[f"f1_{c}"] for c in SUPER_CLASSES},
                **{f"test_auc_{c}": mte[f"auc_{c}"] for c in SUPER_CLASSES},
                **{f"val_f1_{c}": mva[f"f1_{c}"] for c in SUPER_CLASSES})
    curva_df.to_csv(OUT_DIR / "historiales" / f"{id_}.csv", index=False)
    model.save_weights(OUT_DIR / "modelos" / f"{id_}.weights.h5")
    res = resultados()
    pd.concat([res, pd.DataFrame([fila])], ignore_index=True).to_csv(OUT_DIR / "resultados.csv", index=False)
    print(f"--> {id_}: val AUC {mva['auc']:.4f} | val F1 {mva['f1']:.4f} | brecha {fila['brecha']:.3f} | "
          f"{minutos:.1f} min, mejor época {mejor_ep}")
    return fila


def cfg_de(fila):
    out = {k: fila[k] for k in CLAVES}
    for k in ("lr", "gamma", "drop", "l2", "suavizado", "weight_decay"):
        out[k] = float(out[k])
    for k in ("batch", "max_epocas", "paciencia", "semilla"):
        out[k] = int(out[k])
    return out


def elegir(r, margen=0.002):
    """Mejor AUC de validación; entre las que quedan a menos de 'margen', la de menor brecha."""
    r = r.sort_values("val_auc", ascending=False)
    cand = r[r["val_auc"] >= r["val_auc"].iloc[0] - margen]
    return cand.sort_values("brecha").iloc[0].to_dict()


def mejor_de(etapa, rep):
    res = resultados()
    return elegir(res[(res["etapa"] == etapa) & (res["representacion"] == rep)])


def etapa6_arquitecturas():
    for rep in ("espectrograma", "senal"):
        for nombre, (r, desc, _) in ARQUITECTURAS.items():
            if r == rep:
                entrenar(f"A6_{nombre}", "6_arquitecturas", desc, arquitectura=nombre)
    for rep in ("espectrograma", "senal"):
        b = mejor_de("6_arquitecturas", rep)
        print(f"Mejor arquitectura sobre {rep} (validación): {b['descripcion']} (AUC {b['val_auc']:.4f})")


VARIANTES = [
    ("R0", "Sin cambios", {}),
    ("R1", "Más dropout (×1,5) y L2 = 0,001", dict(drop=1.5, l2=1e-3)),
    ("R2", "Aumento de datos", dict(aumento="si")),
    ("R3", "Suavizado de etiquetas (0,1)", dict(suavizado=0.1)),
    ("R4", "AdamW (weight decay 0,0001)", dict(optimizador="adamw", weight_decay=1e-4)),
    ("R5", "Combinada: R1 + R2 + R3", dict(drop=1.5, l2=1e-3, aumento="si", suavizado=0.1)),
    ("R6", "Combinada + parada temprana por pérdida de validación",
     dict(drop=1.5, l2=1e-3, aumento="si", suavizado=0.1, monitor="val_loss")),
]


def etapa7_sobreajuste():
    for rep in ("espectrograma", "senal"):
        base = cfg_de(mejor_de("6_arquitecturas", rep))
        for cod, desc, cambios in VARIANTES:
            entrenar(f"A7_{rep}_{cod}", "7_sobreajuste", desc, **{**base, **cambios})
    for rep in ("espectrograma", "senal"):
        b = mejor_de("7_sobreajuste", rep)
        print(f"Configuración final sobre {rep}: {b['descripcion']} (AUC {b['val_auc']:.4f}, brecha {b['brecha']:.3f})")


def etapa8_semillas(semillas=(42, 7, 123)):
    for rep in ("espectrograma", "senal"):
        final = cfg_de(mejor_de("7_sobreajuste", rep))
        for s in semillas:
            entrenar(f"A8_{rep}_s{s}", f"8_semillas_{rep}", "Configuración final", **{**final, "semilla": s})


def ejecutar_todo():
    etapa6_arquitecturas()
    etapa7_sobreajuste()
    etapa8_semillas()
    tablas_y_figuras()


def _rejilla_curvas(filas, titulo, fichero, columnas=4):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    TR, VA, OR = "#9aa3ab", "#1f5a7a", "#e07b2a"
    n = len(filas)
    nf = int(np.ceil((n + 1) / columnas))
    fig, axs = plt.subplots(nf, columnas, figsize=(3.2 * columnas, 2.5 * nf))
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, f) in zip(axs, filas.iterrows()):
        h = OUT_DIR / "historiales" / f"{f['reutilizado_de'] or f['id']}.csv"
        if not h.exists():
            ax.axis("off"); continue
        c = pd.read_csv(h)
        ax.plot(c["epoca"], c["loss"], color=TR, lw=1.6)
        ax.plot(c["epoca"], c["val_loss"], color=VA, lw=1.6)
        ax.axvline(int(f["mejor_epoca"]), color=OR, lw=1, ls="--")
        ax.set_title(f"{f['descripcion']}"[:46], fontsize=8, loc="left")
        ax.text(0.98, 0.95, f"AUC {f['val_auc']:.3f}\nbrecha {f['brecha']:.3f}".replace(".", ","),
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eee", lw=0.6)
    for ax in axs[n:]:
        ax.axis("off")
    axs[n].legend(handles=[Line2D([], [], color=TR, lw=1.6, label="Pérdida de entrenamiento"),
                           Line2D([], [], color=VA, lw=1.6, label="Pérdida de validación"),
                           Line2D([], [], color=OR, lw=1, ls="--", label="Época cuyos pesos se restauran")],
                  loc="center", frameon=False, fontsize=8)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figuras" / fichero, dpi=200)
    plt.close(fig)


def tablas_y_figuras():
    """Crea las tablas (Excel) y las figuras para la memoria a partir de resultados.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = resultados()
    if not len(res):
        print("Todavía no hay resultados.")
        return
    res["reutilizado_de"] = res["reutilizado_de"].fillna("")
    cols = {"descripcion": "Configuración", "representacion": "Entrada", "parametros": "Parámetros",
            "mejor_epoca": "Mejor época", "minutos": "Minutos", "val_auc": "AUC val.", "val_f1_opt": "F1 val.",
            "val_f1_HYP": "F1 HYP val.", "brecha": "Brecha val.-entren. (sin dropout)", "epoca_min_val_loss": "Época mín. pérdida val.",
            "exceso_val_loss": "Exceso sobre mín. pérdida val.", "subida_final": "Subida final pérdida val."}
    tablas = {}
    for etapa, hoja in [("6_arquitecturas", "6 Arquitecturas"), ("7_sobreajuste", "7 Sobreajuste")]:
        r = res[res["etapa"] == etapa]
        if len(r):
            tablas[hoja] = r[list(cols)].rename(columns=cols).sort_values(["Entrada", "AUC val."], ascending=[True, False]).round(3)
    sem = res[res["etapa"].str.startswith("8_semillas")]
    if len(sem):
        g = sem.groupby("etapa")[["test_auc", "test_f1", "test_precision", "test_recall", "test_f1_HYP", "brecha"]].agg(["mean", "std"])
        g.columns = [f"{a} ({'media' if b == 'mean' else 'desv.'})" for a, b in g.columns]
        tablas["8 Semillas (test)"] = g.reset_index().round(3)
    todo = res[["id", "etapa", "descripcion", "representacion", "parametros", "val_auc", "val_f1_opt", "brecha",
                "test_auc", "test_f1", "test_precision", "test_recall", "test_mae"] + [f"test_f1_{c}" for c in SUPER_CLASSES]]
    tablas["Todos (incl. test)"] = todo.round(3)
    with pd.ExcelWriter(OUT_DIR / "tablas_arquitecturas.xlsx") as w:
        for hoja, t in tablas.items():
            t.to_excel(w, sheet_name=hoja[:31], index=False)

    r6 = res[res["etapa"] == "6_arquitecturas"]
    for rep, nombre in (("espectrograma", "espectrograma"), ("senal", "señal temporal")):
        r = r6[r6["representacion"] == rep]
        if len(r):
            _rejilla_curvas(r, f"Curvas de pérdida: arquitecturas sobre el {nombre}", f"curvas_arquitecturas_{rep}.png")
        r = res[(res["etapa"] == "7_sobreajuste") & (res["representacion"] == rep)]
        if len(r):
            _rejilla_curvas(r, f"Curvas de pérdida: técnicas contra el sobreajuste ({nombre})", f"curvas_sobreajuste_{rep}.png")

    if len(r6):
        r = r6.sort_values("val_auc")
        fig, ax = plt.subplots(1, 3, figsize=(13, 0.42 * len(r) + 1.4), sharey=True)
        y = np.arange(len(r))
        col = ["#1f5a7a" if v == "espectrograma" else "#e07b2a" for v in r["representacion"]]
        for a, cname, tit in zip(ax, ["val_auc", "val_f1_opt", "brecha"],
                                 ["AUC macro (validación)", "F1 macro (validación)", "Brecha pérdida val. − entren."]):
            a.barh(y, r[cname], color=col)
            a.set_title(tit, loc="left", fontsize=10)
            for yy, v in zip(y, r[cname]):
                a.text(v, yy, f" {v:.3f}".replace(".", ","), va="center", fontsize=8)
            a.spines[["top", "right"]].set_visible(False)
        ax[0].set_yticks(y); ax[0].set_yticklabels(r["descripcion"], fontsize=8)
        ax[0].set_xlim(max(0.5, r["val_auc"].min() - 0.03), min(1, r["val_auc"].max() + 0.015))
        ax[1].set_xlim(max(0, r["val_f1_opt"].min() - 0.05), min(1, r["val_f1_opt"].max() + 0.03))
        fig.tight_layout(); fig.savefig(OUT_DIR / "figuras" / "comparacion_arquitecturas.png", dpi=200); plt.close(fig)

    print("Tablas en", (OUT_DIR / "tablas_arquitecturas.xlsx").resolve())
    print("Figuras en", (OUT_DIR / "figuras").resolve())
    return tablas
