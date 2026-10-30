# Modelos propios en PyTorch. Todos reciben
#   X  (B, L, 1 + k)  y normalizada en la columna 0 y calendario de las horas de entrada
#   Xf (B, H, k)      calendario de las horas a pronosticar (se conoce de antemano)
#   y  (B, H)         solo en entrenamiento, para teacher forcing
# y devuelven (B, H) en la misma escala normalizada.

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    # Multi-salida, aplana la ventana de y y el calendario futuro y saca los H pasos juntos
    def __init__(self, L, H, k, oculto=512, capas=2, dropout=0.1):
        super().__init__()
        dims = [L + H * k] + [oculto] * capas
        bloques = []
        for a, b in zip(dims[:-1], dims[1:]):
            bloques += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.red = nn.Sequential(*bloques, nn.Linear(dims[-1], H))

    def forward(self, X, Xf, y=None):
        return self.red(torch.cat([X[:, :, 0], Xf.flatten(1)], dim=1))


class DLinear(nn.Module):
    # Zeng et al. (2023). Separa tendencia (media móvil) y resto, y a cada parte le aplica
    # una capa lineal de L a H. Solo usa la serie, sin covariables.
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.lin_tendencia = nn.Linear(L, H)
        self.lin_estacional = nn.Linear(L, H)

    def tendencia(self, x):
        a = (self.kernel - 1) // 2
        b = self.kernel - 1 - a
        xp = torch.cat([x[:, :1].expand(-1, a), x, x[:, -1:].expand(-1, b)], dim=1)
        return F.avg_pool1d(xp.unsqueeze(1), self.kernel, stride=1).squeeze(1)

    def forward(self, X, Xf, y=None):
        x = X[:, :, 0]
        t = self.tendencia(x)
        return self.lin_tendencia(t) + self.lin_estacional(x - t)


class NLinear(nn.Module):
    # Resta el último valor de la ventana, aplica una lineal de L a H y lo vuelve a sumar
    def __init__(self, L, H):
        super().__init__()
        self.lin = nn.Linear(L, H)

    def forward(self, X, Xf, y=None):
        x = X[:, :, 0]
        ultimo = x[:, -1:]
        return self.lin(x - ultimo) + ultimo


class LSTMMultiSalida(nn.Module):
    # La LSTM lee la ventana y una capa densa devuelve los H pasos de una vez.
    # El calendario futuro entra a la capa densa junto con el último estado oculto.
    def __init__(self, L, H, k, oculto=128, capas=2, dropout=0.1, celda="lstm"):
        super().__init__()
        rnn = nn.LSTM if celda == "lstm" else nn.GRU
        self.rnn = rnn(1 + k, oculto, capas, batch_first=True,
                       dropout=dropout if capas > 1 else 0.0)
        self.cabeza = nn.Sequential(nn.Linear(oculto + H * k, oculto), nn.ReLU(),
                                    nn.Linear(oculto, H))

    def forward(self, X, Xf, y=None):
        salida, _ = self.rnn(X)
        return self.cabeza(torch.cat([salida[:, -1], Xf.flatten(1)], dim=1))


class LSTMAutorregresiva(nn.Module):
    # Predice un paso y realimenta su predicción. En el paso t recibe y_t y el calendario
    # de t+1 y devuelve y_{t+1}. En entrenamiento usa teacher forcing (los valores reales).
    def __init__(self, L, H, k, oculto=128, capas=2, dropout=0.1):
        super().__init__()
        self.L, self.H = L, H
        self.rnn = nn.LSTM(1 + k, oculto, capas, batch_first=True,
                           dropout=dropout if capas > 1 else 0.0)
        self.cabeza = nn.Linear(oculto, 1)

    def forward(self, X, Xf, y=None):
        y_in, cal_in = X[:, :, :1], X[:, :, 1:]
        cal_sig = torch.cat([cal_in[:, 1:], Xf[:, :1]], dim=1)
        entrada = torch.cat([y_in, cal_sig], dim=-1)
        if self.training and y is not None:
            dec = torch.cat([y[:, :-1, None], Xf[:, 1:]], dim=-1)
            salida, _ = self.rnn(torch.cat([entrada, dec], dim=1))
            return self.cabeza(salida[:, self.L - 1:]).squeeze(-1)
        salida, estado = self.rnn(entrada)
        p = self.cabeza(salida[:, -1])
        preds = [p]
        for j in range(1, self.H):
            salida, estado = self.rnn(torch.cat([p, Xf[:, j]], dim=-1).unsqueeze(1), estado)
            p = self.cabeza(salida[:, -1])
            preds.append(p)
        return torch.cat(preds, dim=1)


class Seq2Seq(nn.Module):
    # Encoder-decoder. El encoder resume la ventana y el decoder genera los H pasos
    # recibiendo en cada paso el calendario de esa hora y el valor anterior.
    # En entrenamiento, con probabilidad teacher_forcing por lote, el decoder recibe
    # los valores reales en vez de sus propias predicciones.
    def __init__(self, L, H, k, oculto=128, capas=2, dropout=0.1, teacher_forcing=0.5):
        super().__init__()
        self.H, self.tf = H, teacher_forcing
        d = dropout if capas > 1 else 0.0
        self.encoder = nn.LSTM(1 + k, oculto, capas, batch_first=True, dropout=d)
        self.decoder = nn.LSTM(1 + k, oculto, capas, batch_first=True, dropout=d)
        self.cabeza = nn.Linear(oculto, 1)

    def forward(self, X, Xf, y=None):
        _, estado = self.encoder(X)
        prev = X[:, -1, :1]
        if self.training and y is not None and random.random() < self.tf:
            anteriores = torch.cat([prev, y[:, :-1]], dim=1).unsqueeze(-1)
            salida, _ = self.decoder(torch.cat([anteriores, Xf], dim=-1), estado)
            return self.cabeza(salida).squeeze(-1)
        preds = []
        for j in range(self.H):
            salida, estado = self.decoder(torch.cat([prev, Xf[:, j]], dim=-1).unsqueeze(1), estado)
            prev = self.cabeza(salida[:, -1])
            preds.append(prev)
        return torch.cat(preds, dim=1)


class RevIN(nn.Module):
    # Kim et al. (2022). Normaliza cada ventana con su propia media y desviación,
    # el modelo trabaja en esa escala y al final se deshace la normalización.
    # Ayuda cuando el nivel de la serie cambia entre entrenamiento y prueba.
    def __init__(self, modelo, eps=1e-5):
        super().__init__()
        self.modelo = modelo
        self.peso = nn.Parameter(torch.ones(1))
        self.sesgo = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, X, Xf, y=None):
        media = X[:, :, :1].mean(dim=1, keepdim=True)
        desv = X[:, :, :1].std(dim=1, keepdim=True) + self.eps
        yn = (X[:, :, :1] - media) / desv * self.peso + self.sesgo
        Xn = torch.cat([yn, X[:, :, 1:]], dim=-1)
        y_tf = None if y is None else (y - media[:, :, 0]) / desv[:, :, 0] * self.peso + self.sesgo
        salida = self.modelo(Xn, Xf, y_tf)
        return (salida - self.sesgo) / self.peso * desv[:, :, 0] + media[:, :, 0]


def construir(cfg, k):
    L, H = cfg["L"], cfg["H"]
    nombre = cfg["modelo"]
    oc, ca, dr = cfg.get("oculto", 128), cfg.get("capas", 2), cfg.get("dropout", 0.1)
    if nombre == "mlp":
        m = MLP(L, H, k, oc, ca, dr)
    elif nombre == "dlinear":
        m = DLinear(L, H, cfg.get("kernel", 25))
    elif nombre == "nlinear":
        m = NLinear(L, H)
    elif nombre in ("lstm", "gru"):
        m = LSTMMultiSalida(L, H, k, oc, ca, dr, celda=nombre)
    elif nombre == "lstm_ar":
        m = LSTMAutorregresiva(L, H, k, oc, ca, dr)
    elif nombre == "seq2seq":
        m = Seq2Seq(L, H, k, oc, ca, dr, cfg.get("teacher_forcing", 0.5))
    else:
        raise ValueError(nombre)
    if cfg.get("norm") == "revin":
        m = RevIN(m)
    return m


def contar_parametros(m):
    total = sum(p.numel() for p in m.parameters())
    entrenables = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, entrenables
