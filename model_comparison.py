import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
import time

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import BorderlineSMOTE

import xgboost as xgb
import lightgbm as lgb

from drug_food_interaction_pipeline import load_all_datasets, build_feature_matrix, apply_rfe

SEED = 42
np.random.seed(SEED)


def build_voting_ensemble() -> VotingClassifier:
    """Our champion Soft-Voting Ensemble from the main pipeline."""
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        reg_alpha=0.5, reg_lambda=1.5, subsample=0.75,
        colsample_bytree=0.6, min_child_weight=5, gamma=0.3,
        use_label_encoder=False, eval_metric="mlogloss",
        objective="multi:softprob", num_class=3,
        random_state=SEED, verbosity=0,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.08,
        subsample=0.75, min_samples_leaf=5, random_state=SEED,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf), ("gb", gb_clf)],
        voting="soft", weights=[2, 2, 1],
    )


def get_all_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["ROC AUC"] = round(roc_auc_score(y_true, y_prob, multi_class="ovr"), 4)
        except Exception:
            metrics["ROC AUC"] = float("nan")
    return metrics


def main():
    save_dir = "./outputs/comparison_results"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("   ML Model Benchmarking Suite — Real-World FDA Dataset")
    print("=" * 60)

    # ── 1. Load real-world clinical data ────────────────────────────────
    print("\n[Step 1] Loading real-world FDA dataset with native SMILES...")
    df_all = load_all_datasets("clinical_interaction_real.csv")
    df       = df_all[df_all["split"] == "train"].copy()
    unseen_df = df_all[df_all["split"] == "unseen"].copy()
    print(f"  Train: {len(df)} rows | Unseen (LODO): {len(unseen_df)} rows")

    # ── 2. Feature engineering ──────────────────────────────────────────
    print("\n[Step 2] Building molecular feature matrices...")
    feat_df      = build_feature_matrix(df, fp_bits=256)
    unseen_feat  = build_feature_matrix(unseen_df, fp_bits=256)
    feat_names   = feat_df.columns.tolist()

    X         = feat_df.values
    y_cls     = df["risk_level"].values
    X_unseen  = unseen_feat.values
    y_unseen  = unseen_df["risk_level"].values

    # ── 3. Split, scale, RFE ────────────────────────────────────────────
    print("\n[Step 3] Train/Val split, scaling, RFE...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cls, test_size=0.2, random_state=SEED, stratify=y_cls
    )
    scaler = StandardScaler()
    X_train    = scaler.fit_transform(X_train)
    X_val      = scaler.transform(X_val)
    X_unseen_s = scaler.transform(X_unseen)

    n_fp = 512
    rfe_mask = apply_rfe(X_train[:, n_fp:], y_train, n_features=20)
    X_tr  = np.hstack([X_train[:, :n_fp], X_train[:, n_fp:][:, rfe_mask]])
    X_v   = np.hstack([X_val[:, :n_fp],   X_val[:, n_fp:][:, rfe_mask]])
    X_uns = np.hstack([X_unseen_s[:, :n_fp], X_unseen_s[:, n_fp:][:, rfe_mask]])

    # ── 4. BorderlineSMOTE ──────────────────────────────────────────────
    print("\n[Step 4] Applying BorderlineSMOTE to training set...")
    smote = BorderlineSMOTE(random_state=SEED, kind="borderline-1")
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_train)
    print(f"  Expanded: {len(y_train)} → {len(y_tr_sm)} rows")

    # ── 5. Define all models ────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            solver="lbfgs", random_state=SEED
        ),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, metric="minkowski"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=SEED
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            lambda_l1=0.5, lambda_l2=1.5, subsample=0.75,
            colsample_bytree=0.6, min_child_samples=10,
            class_weight="balanced", random_state=SEED, verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=1.5, subsample=0.75,
            colsample_bytree=0.6, min_child_weight=5, gamma=0.3,
            use_label_encoder=False, eval_metric="mlogloss",
            objective="multi:softprob", num_class=3,
            random_state=SEED, verbosity=0
        ),
        "Soft-Voting Ensemble\n(XGB+RF+GB)": build_voting_ensemble(),
    }

    # ── 6. Benchmark ────────────────────────────────────────────────────
    print("\n[Step 5] Benchmarking all models...")
    results = []

    for name, model in models.items():
        display = name.replace("\n", " ")
        print(f"  ▸ Training {display}...")
        t0 = time.time()
        model.fit(X_tr_sm, y_tr_sm)
        elapsed = time.time() - t0

        has_proba = hasattr(model, "predict_proba")
        p_tr  = model.predict(X_tr)
        p_val = model.predict(X_v)
        p_uns = model.predict(X_uns)

        prob_tr   = model.predict_proba(X_tr)  if has_proba else None
        prob_val  = model.predict_proba(X_v)   if has_proba else None
        prob_uns  = model.predict_proba(X_uns) if has_proba else None

        m_tr  = get_all_metrics(y_train, p_tr,  prob_tr)
        m_val = get_all_metrics(y_val,   p_val, prob_val)
        m_uns = get_all_metrics(y_unseen, p_uns, prob_uns)

        # Compute confusion matrix for unseen
        cm = confusion_matrix(y_unseen, p_uns, labels=[0, 1, 2])

        # Save individual text report
        safe_name = display.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "+")
        txt = f"{'='*55}\n  Model: {display}\n{'='*55}\n\n"
        txt += f"  Training Time : {elapsed:.2f}s\n\n"
        for split_name, mdict in [("Train", m_tr), ("Validation", m_val), ("Unseen (LODO)", m_uns)]:
            txt += f"[{split_name}]\n"
            for k, v in mdict.items():
                txt += f"  {k:22s}: {v:.4f}\n"
            txt += "\n"
        txt += f"[Unseen Confusion Matrix (Neutral|Moderate|Critical)]\n{cm}\n"

        with open(f"{save_dir}/{safe_name}_metrics.txt", "w") as f:
            f.write(txt)

        results.append({
            "Model":               display,
            "Train Acc":           m_tr["Accuracy"],
            "Val Acc":             m_val["Accuracy"],
            "Unseen Acc":          m_uns["Accuracy"],
            "Val F1":              m_val["F1 Score"],
            "Unseen F1":           m_uns["F1 Score"],
            "Val AUC":             m_val.get("ROC AUC", float("nan")),
            "Unseen AUC":          m_uns.get("ROC AUC", float("nan")),
        })
        print(f"    Train={m_tr['Accuracy']:.3f}  Val={m_val['Accuracy']:.3f}  Unseen={m_uns['Accuracy']:.3f}")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Val Acc", ascending=False)
    df_res.to_csv(f"{save_dir}/model_benchmarks.csv", index=False)

    # ── 7. Rich comparison visualizations ──────────────────────────────
    print("\n[Step 6] Generating comparison visualizations...")

    labels      = [m.replace("\n", "\n") for m in df_res["Model"]]
    x           = np.arange(len(labels))
    w           = 0.22
    colors      = {"Train": "#4e79a7", "Val": "#f28e2b", "Unseen": "#59a14f"}
    highlight   = "#e15759"  # colour for our ensemble

    # ── Plot 1: Accuracy comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    b1 = ax.bar(x - w, df_res["Train Acc"],  w, label="Train",      color=colors["Train"],  alpha=0.9)
    b2 = ax.bar(x,     df_res["Val Acc"],    w, label="Validation", color=colors["Val"],    alpha=0.9)
    b3 = ax.bar(x + w, df_res["Unseen Acc"], w, label="Unseen (LODO)", color=colors["Unseen"], alpha=0.9)
    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color="white", fontweight="bold")
    ax.axhline(0.90, color=highlight, linestyle="--", linewidth=1.5, alpha=0.8, label="90% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", color="white", fontsize=12)
    ax.set_title("Model Accuracy Comparison — Real-World FDA Drug-Food Interactions",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333366")
    legend = ax.legend(loc="upper right", framealpha=0.3, labelcolor="white",
                       facecolor="#16213e", edgecolor="#333366", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_comparison_all_models.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # ── Plot 2: F1 Score comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    b1 = ax.bar(x - w/2, df_res["Val F1"],    w, label="Val F1",         color=colors["Val"],    alpha=0.9)
    b2 = ax.bar(x + w/2, df_res["Unseen F1"], w, label="Unseen F1 (LODO)", color=colors["Unseen"], alpha=0.9)
    for bar in [b1, b2]:
        for rect in bar:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color="white", fontweight="bold")
    ax.axhline(0.90, color=highlight, linestyle="--", linewidth=1.5, alpha=0.8, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Weighted F1 Score", color="white", fontsize=12)
    ax.set_title("Model F1 Score Comparison (Weighted) — Validation vs Unseen",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333366")
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white",
              facecolor="#16213e", edgecolor="#333366", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/f1_comparison_all_models.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # ── Plot 3: Heatmap of all metrics ───────────────────────────────────
    heat_cols = ["Train Acc", "Val Acc", "Unseen Acc", "Val F1", "Unseen F1", "Val AUC", "Unseen AUC"]
    heat_df   = df_res.set_index("Model")[heat_cols].astype(float)
    fig, ax   = plt.subplots(figsize=(13, len(labels) * 0.7 + 2))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0.5, vmax=1.0, ax=ax, cbar_kws={"shrink": 0.6},
                linewidths=0.5, linecolor="#1a1a2e")
    ax.set_title("All-Model Performance Heatmap — Real-World FDA Dataset",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white", axis="both", labelsize=9)
    ax.set_xticklabels(heat_cols, rotation=35, ha="right", color="white")
    ax.set_yticklabels(heat_df.index, rotation=0, color="white")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_heatmap_all_models.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # ── Plot 4: ROC AUC scatter for Val vs Unseen ────────────────────────
    valid_auc = df_res.dropna(subset=["Val AUC", "Unseen AUC"])
    fig, ax   = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    sc = ax.scatter(valid_auc["Val AUC"], valid_auc["Unseen AUC"],
                    c=valid_auc["Val Acc"], cmap="plasma",
                    s=200, zorder=5, edgecolors="white", linewidths=0.8)
    for _, row in valid_auc.iterrows():
        ax.annotate(row["Model"].replace("\n", " "), (row["Val AUC"], row["Unseen AUC"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color="white", alpha=0.9)
    ax.plot([0.5, 1.0], [0.5, 1.0], "w--", alpha=0.4, lw=1.2, label="Perfect parity line")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Validation Accuracy", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("Validation ROC AUC", color="white", fontsize=11)
    ax.set_ylabel("Unseen ROC AUC (LODO)", color="white", fontsize=11)
    ax.set_title("Generalisation Scatter: Val AUC vs Unseen AUC",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#333366")
    ax.legend(fontsize=9, labelcolor="white", facecolor="#16213e",
              edgecolor="#333366", framealpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_auc_scatter_val_vs_unseen.png", dpi=180,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── 8. Markdown report ───────────────────────────────────────────────
    best_row = df_res.iloc[0]
    md  = "# ML Model Benchmarking Report — Real-World FDA Drug-Food Interactions\n\n"
    md += "> Dataset: `clinical_interaction_real.csv` | Split: Leave-One-Drug-Out (LODO)\n"
    md += "> Preprocessing: BorderlineSMOTE | RFE(20) | StandardScaler | Morgan FP (256-bit)\n\n"
    md += "## Performance Summary\n\n"
    md += df_res.to_markdown(index=False)
    md += f"\n\n## Winner\n**{best_row['Model']}** achieved the highest validation accuracy at "
    md += f"`{best_row['Val Acc']:.4f}` and unseen accuracy at `{best_row['Unseen Acc']:.4f}`.\n\n"
    md += "## Outputs\n"
    md += "- `accuracy_comparison_all_models.png` — Bar chart per model across Train/Val/Unseen\n"
    md += "- `f1_comparison_all_models.png` — F1 Score comparison\n"
    md += "- `performance_heatmap_all_models.png` — All-metric heatmap\n"
    md += "- `roc_auc_scatter_val_vs_unseen.png` — Generalisation scatter plot\n"
    md += "- `<model>_metrics.txt` — Per-model text report\n"
    md += "- `model_benchmarks.csv` — Full numeric table\n"

    with open(f"{save_dir}/comparison_report.md", "w") as f:
        f.write(md)

    print(f"\n{'='*60}")
    print(f"  Benchmarking complete! Top 3 Models by Validation Accuracy:")
    print(f"{'='*60}")
    for i, row in df_res.head(3).iterrows():
        print(f"  {row['Model']:35s}  Val={row['Val Acc']:.3f}  Unseen={row['Unseen Acc']:.3f}")
    print(f"\n  All outputs saved to: {save_dir}/\n")


if __name__ == "__main__":
    main()
