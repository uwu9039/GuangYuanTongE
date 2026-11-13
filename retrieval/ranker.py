"""学习排序器：numpy 逻辑回归（离线训练，权重落 JSON）。

不引入 sklearn / torch：目标机的 `requirements.txt` 只有 fastapi/uvicorn/numpy，而这
个模型只有几十个特征，牛顿法（IRLS）五轮就收敛，训练一次几秒。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


@dataclass
class Ranker:
    w: np.ndarray
    b: float
    mean: np.ndarray
    std: np.ndarray
    names: tuple[str, ...] = FEATURE_NAMES

    def score(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        z = (x - self.mean) / np.maximum(self.std, 1e-6)
        return (z @ self.w + self.b).astype(np.float32)

    def prob(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(x), -30, 30)))

    def top_features(self, k: int = 12) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.w))[:k]
        return [(self.names[i], round(float(self.w[i]), 4)) for i in order]

    # ------------------------------------------------------------------ train

    @classmethod
    def train(cls, x: np.ndarray, y: np.ndarray, l2: float = 1.0,
              iters: int = 12) -> "Ranker":
        """L2 正则逻辑回归，牛顿法（IRLS）。

        `x` 是 (N, F) 特征矩阵，`y` 是 0/1 相关标签。样本里正负比悬殊，按标签重加权
        到各占一半，否则模型会把所有候选都判成不相关。
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        z = (x - mean) / std
        z = np.hstack([z, np.ones((z.shape[0], 1))])

        n_pos = max(y.sum(), 1.0)
        n_neg = max(len(y) - y.sum(), 1.0)
        sw = np.where(y > 0.5, 0.5 / n_pos, 0.5 / n_neg)

        beta = np.zeros(z.shape[1])
        reg = np.eye(z.shape[1]) * l2
        reg[-1, -1] = 0.0  # 截距不正则
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -30, 30)))
            g = z.T @ (sw * (p - y)) + reg @ beta
            wgt = sw * p * (1.0 - p)
            H = (z * wgt[:, None]).T @ z + reg + np.eye(z.shape[1]) * 1e-8
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w=beta[:-1].astype(np.float32), b=float(beta[-1]),
                   mean=mean.astype(np.float32), std=std.astype(np.float32))

    # ------------------------------------------------------------------- io

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "names": list(self.names),
            "w": [round(float(v), 6) for v in self.w],
            "b": float(self.b),
            "mean": [round(float(v), 6) for v in self.mean],
            "std": [round(float(v), 6) for v in self.std],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ranker":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        names = tuple(d.get("names") or FEATURE_NAMES)
        if names != FEATURE_NAMES:
            raise ValueError("排序器特征版本与代码不一致，需要重新训练")
        return cls(w=np.asarray(d["w"], dtype=np.float32), b=float(d["b"]),
                   mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32), names=names)
