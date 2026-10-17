"""RUL 预测模型：经典 ML 基线与一维 CNN。

统一约定：
- 目标值为 log(cycle_life)，预测后取 exp 回到循环数（寿命跨度 148~2237，对数域更稳）；
- 特征先做中位数填补 + 标准化；
- 所有随机过程固定种子，保证可复现。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "median_ape": float(np.median(ape) * 100),
        "mean_ape": float(np.mean(ape) * 100),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "max_ape": float(np.max(ape) * 100),
        "n": int(len(y_true)),
    }


def make_elasticnet(alpha: float = 0.03, l1_ratio: float = 0.5) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, random_state=SEED)),
    ])


def make_rf(n_estimators: int = 400, max_depth=None) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=1,
            random_state=SEED, n_jobs=-1)),
    ])


def make_gbm(n_estimators: int = 300, learning_rate: float = 0.03, max_depth: int = 2) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.9, random_state=SEED)),
    ])


def make_mlp(hidden=(64, 32), alpha: float = 3.0) -> Pipeline:
    """神经网络基线（小样本下比 CNN 更稳，但仍不如树模型）。"""
    from sklearn.neural_network import MLPRegressor

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=4000,
                           learning_rate_init=3e-3, random_state=SEED)),
    ])


# ---------------------------------------------------------------------------
# 1-D CNN
# ---------------------------------------------------------------------------
def _torch():
    import torch  # 延迟导入，没装 torch 也能跑经典模型
    return torch


class DQCNN:
    """3 通道 ΔQ(V) 曲线 + 标量特征的轻量 1D CNN。

    小样本（41 颗）下的关键设计：
    - 5 折内部交叉验证训练 + 早停（用验证折挑最优 epoch），再以折模型集成预测；
    - 曲线高斯增广、权重衰减、Dropout 抑制过拟合；
    - 多随机种子集成降低方差。
    这样既不接触测试集，也避免单模型在全批上「背答案」。
    """

    def __init__(self, n_scalar: int, n_seed: int = 2, epochs: int = 400,
                 lr: float = 2e-3, patience: int = 40, n_folds: int = 5):
        self.n_scalar = n_scalar
        self.n_seed = n_seed
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.n_folds = n_folds
        self.models = []
        self.curve_mu = None
        self.curve_sd = None
        self.sc_mu = None
        self.sc_sd = None

    def _build(self, torch):
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self, n_scalar):
                super().__init__()
                self.c1 = nn.Conv1d(3, 12, 9, padding=4)
                self.b1 = nn.BatchNorm1d(12)
                self.c2 = nn.Conv1d(12, 24, 5, padding=2)
                self.b2 = nn.BatchNorm1d(24)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(24 + n_scalar, 32)
                self.do = nn.Dropout(0.3)
                self.fc2 = nn.Linear(32, 1)
                self.act = nn.ReLU()

            def forward(self, xc, xs):
                h = self.act(self.b1(self.c1(xc)))
                h = nn.functional.max_pool1d(h, 2)
                h = self.act(self.b2(self.c2(h)))
                h = self.pool(h).flatten(1)
                h = torch.cat([h, xs], dim=1)
                h = self.do(self.act(self.fc1(h)))
                return self.fc2(h).squeeze(-1)

        return Net(self.n_scalar)

    def _train_one(self, torch, Xc, Xs, yl, val_idx, seed):
        net = self._build(torch)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        tr = np.array([i for i in range(len(yl)) if i not in set(val_idx)])
        xc = torch.tensor(Xc[tr], dtype=torch.float32)
        xs = torch.tensor(Xs[tr], dtype=torch.float32)
        yt = torch.tensor(yl[tr], dtype=torch.float32)
        xv = torch.tensor(Xc[val_idx], dtype=torch.float32)
        xsv = torch.tensor(Xs[val_idx], dtype=torch.float32)
        yv = torch.tensor(yl[val_idx], dtype=torch.float32)

        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            net.train()
            aug = torch.randn_like(xc) * 0.05
            out = net(xc + aug, xs)
            loss = torch.nn.functional.mse_loss(out, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if ep % 5 == 0:
                net.eval()
                with torch.no_grad():
                    vloss = torch.nn.functional.mse_loss(net(xv, xsv), yv).item()
                if vloss < best - 1e-5:
                    best, bad = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience // 5:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, curves: np.ndarray, scalars: np.ndarray, y: np.ndarray):
        torch = _torch()
        self.curve_mu = np.nanmean(curves, axis=(0, 2), keepdims=True)
        self.curve_sd = np.nanstd(curves, axis=(0, 2), keepdims=True) + 1e-6
        self.sc_mu = np.nanmean(scalars, axis=0)
        self.sc_sd = np.nanstd(scalars, axis=0) + 1e-6
        Xc = np.nan_to_num((curves - self.curve_mu) / self.curve_sd, nan=0.0)
        Xs = (np.nan_to_num(scalars, nan=0.0) - self.sc_mu) / self.sc_sd
        yl = np.log(y)

        n = len(yl)
        idx = np.arange(n)
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        folds = np.array_split(idx, self.n_folds)

        self.models = []
        for k, val_idx in enumerate(folds):
            for s in range(self.n_seed):
                torch.manual_seed(SEED + 100 * k + s)
                np.random.seed(SEED + 100 * k + s)
                self.models.append(self._train_one(torch, Xc, Xs, yl, val_idx, s))
        return self

    def predict(self, curves: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        torch = _torch()
        Xc = np.nan_to_num((curves - self.curve_mu) / self.curve_sd, nan=0.0)
        Xs = (np.nan_to_num(scalars, nan=0.0) - self.sc_mu) / self.sc_sd
        xc = torch.tensor(Xc, dtype=torch.float32)
        xs = torch.tensor(Xs, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for net in self.models:
                preds.append(net(xc, xs).numpy())
        return np.exp(np.mean(preds, axis=0))
