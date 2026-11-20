"""Metricas, tabelas e graficos do classificador de especies (sem dependencia de PyTorch).

Todas as funcoes recebem rotulos/predicoes como indices 0..K-1 e uma lista
`class_names` de tamanho K na mesma ordem.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PSEUDO_THRESHOLD_NOTE = (
    "O corte de confianca usado para pseudorrotulos (ex.: 0,5) e uma regra operativa; "
    "confianca do softmax nao garante que o rotulo esteja correto.")


def global_metrics(y_true, y_pred, labels) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    present = sorted(set(y_true.tolist()))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # balanced_accuracy considera apenas classes presentes em y_true.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(present) > 1 else None,
        "f1_macro": float(f),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        # macro sobre classes presentes no conjunto (evita penalizar classes ausentes do teste).
        "f1_macro_present_classes": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)) if len(present) > 1 else None,
        "number_of_samples": int(len(y_true)),
        "number_of_classes": int(len(labels)),
        "number_of_classes_present": int(len(present)),
    }


def per_class_metrics(y_true, y_pred, probs, labels, class_names, class_ids) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.sum()
    rows = []
    for k, lab in enumerate(labels):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp; tn = n - tp - fn - fp
        auc = None
        if probs is not None:
            pos = (y_true == lab).astype(int)
            if 0 < pos.sum() < len(pos):
                auc = float(roc_auc_score(pos, np.asarray(probs)[:, k]))
        rows.append({
            "class_index": lab,
            "species_id": class_ids[k],
            "species_name": class_names[k],
            "precision": float(p[k]),
            "recall": float(r[k]),
            "f1": float(f[k]),
            "support": int(s[k]),
            "support_normalized": float(s[k] / n) if n else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "auc_ovr": auc,
        })
    return pd.DataFrame(rows)


def text_report(y_true, y_pred, labels, class_names) -> str:
    return classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                 zero_division=0, digits=4)


def save_confusion(y_true, y_pred, labels, class_names, out_dir: Path, prefix: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        col = np.nan_to_num(cm / cm.sum(axis=0, keepdims=True))
    idx = [f"true:{n}" for n in class_names]; cols = [f"pred:{n}" for n in class_names]
    pd.DataFrame(cm, index=idx, columns=cols).to_csv(out_dir / f"{prefix}_confusion_counts.csv")
    pd.DataFrame(row, index=idx, columns=cols).to_csv(out_dir / f"{prefix}_confusion_row_normalized.csv")
    pd.DataFrame(col, index=idx, columns=cols).to_csv(out_dir / f"{prefix}_confusion_col_normalized.csv")
    for mat, suffix, fmt in ((cm, "counts", "d"), (row, "row_normalized", ".2f"), (col, "col_normalized", ".2f")):
        k = len(class_names)
        fig, ax = plt.subplots(figsize=(1.2 * k + 3, 1.0 * k + 2.5))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(mat.max() if fmt == "d" else 1) or 1)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predito"); ax.set_ylabel("Referencia (anotado)")
        thr = (mat.max() if mat.size else 0) / 2
        for i in range(k):
            for j in range(k):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thr else "black", fontsize=9)
        ax.set_title(f"{title} ({suffix})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_confusion_{suffix}.png", dpi=150); plt.close(fig)
    return cm


def calibration(y_true, probs, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        acc = float(correct[m].mean()) if cnt else None
        mc = float(conf[m].mean()) if cnt else None
        if cnt:
            ece += cnt / len(conf) * abs(acc - mc)
        rows.append({"bin_low": lo, "bin_high": hi, "count": cnt, "accuracy": acc, "mean_confidence": mc})
    return pd.DataFrame(rows), float(ece)


def accuracy_vs_threshold(y_true, probs, thresholds=None) -> pd.DataFrame:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    conf = probs.max(axis=1); correct = probs.argmax(axis=1) == y_true
    thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"threshold": float(t), "coverage": float(m.mean()), "n_kept": int(m.sum()),
                     "accuracy_kept": float(correct[m].mean()) if m.any() else None})
    return pd.DataFrame(rows)


def save_confidence_outputs(y_true, probs, class_names, out_dir: Path, prefix: str) -> dict:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    df = pd.DataFrame({"true_index": y_true, "pred_index": pred, "confidence": conf,
                       "true_name": [class_names[i] for i in y_true],
                       "pred_name": [class_names[i] for i in pred], "correct": pred == y_true})
    df.to_csv(out_dir / f"{prefix}_confidence_per_sample.csv", index=False)
    by_class = df.groupby("true_name")["confidence"].describe()
    by_class.to_csv(out_dir / f"{prefix}_confidence_by_class.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([conf[pred == y_true], conf[pred != y_true]], bins=20, range=(0, 1), stacked=True,
            label=["corretas", "erradas"])
    ax.set_xlabel("Confianca maxima (softmax)"); ax.set_ylabel("Amostras"); ax.legend()
    ax.set_title("Histograma de confianca"); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confidence_histogram.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.1), 4))
    data = [conf[y_true == k] for k in range(len(class_names))]
    ax.boxplot([d if len(d) else [np.nan] for d in data], labels=class_names)
    ax.set_ylabel("Confianca maxima"); ax.set_title("Confianca por especie (referencia)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confidence_by_class.png", dpi=150); plt.close(fig)

    cal, ece = calibration(y_true, probs)
    cal.to_csv(out_dir / f"{prefix}_reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = cal.dropna()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibracao perfeita")
    ax.plot(valid["mean_confidence"], valid["accuracy"], "o-", label=f"modelo (ECE={ece:.3f})")
    ax.set_xlabel("Confianca media"); ax.set_ylabel("Acuracia"); ax.legend(); ax.set_title("Reliability diagram")
    fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_reliability.png", dpi=150); plt.close(fig)

    thr = accuracy_vs_threshold(y_true, probs)
    thr.to_csv(out_dir / f"{prefix}_accuracy_vs_confidence_threshold.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr["threshold"], thr["accuracy_kept"], "o-", label="acuracia (amostras mantidas)")
    ax.plot(thr["threshold"], thr["coverage"], "s--", label="cobertura")
    ax.axvline(0.5, color="red", lw=0.8, label="corte operativo 0,5")
    ax.set_xlabel("Limiar de confianca"); ax.set_ylim(0, 1.02); ax.legend(); ax.set_title(
        "Acuracia x limiar de confianca")
    fig.text(0.01, 0.01, "Corte 0,5 = regra operativa, nao garantia de rotulo correto.", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_accuracy_vs_confidence_threshold.png", dpi=150)
    plt.close(fig)
    return {"ece": ece, "pseudo_threshold_note": PSEUDO_THRESHOLD_NOTE}


def save_training_curves(history: pd.DataFrame, out_dir: Path, class_names=None):
    """Salva cada curva em PNG + CSV com exatamente os dados plotados."""
    history.to_csv(out_dir / "learning_curve_species.csv", index=False)
    specs = [
        ("loss", ["train_loss", "val_loss"], "Loss (CrossEntropy)"),
        ("f1_macro", ["train_f1_macro", "val_f1_macro"], "F1-macro"),
        ("accuracy", ["train_accuracy", "val_accuracy"], "Acuracia"),
        ("learning_rate", ["lr"], "Learning rate"),
    ]
    for name, cols, ylabel in specs:
        cols = [c for c in cols if c in history.columns]
        if not cols:
            continue
        history[["epoch"] + cols].to_csv(out_dir / f"curve_{name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in cols:
            ax.plot(history["epoch"], history[c], "o-", label=c)
        ax.set_xlabel("Epoca"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
        if name == "learning_rate":
            ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(out_dir / f"curve_{name}.png", dpi=150); plt.close(fig)
    per_class = [c for c in history.columns if c.startswith("val_f1_class_")]
    if per_class:
        history[["epoch"] + per_class].to_csv(out_dir / "curve_val_f1_per_class.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in per_class:
            ax.plot(history["epoch"], history[c], "o-", label=c.replace("val_f1_class_", ""))
        ax.set_xlabel("Epoca"); ax.set_ylabel("F1 (validacao)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "curve_val_f1_per_class.png", dpi=150); plt.close(fig)


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
