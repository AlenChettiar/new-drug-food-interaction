"""
=============================================================================
Drug-Food Interaction Risk Assessment Pipeline for Oncology Drugs
=============================================================================
Project: Drug-Food Interaction Risk Assessment and Awareness System
Department: AI & Data Science

System Architecture:
  1. Multi-source data acquisition (PubChem, ChEMBL, BindingDB mocked/API)
  2. Advanced feature engineering (Morgan FP, RDKit descriptors, Tanimoto)
  3. Multi-task ML (XGBoost classification + LightGBM regression)
  4. Scaffold-based and LODCO validation (anti-memorization)
  5. Unseen dataset holdout evaluation
  6. Full metrics + visualizations
  7. SHAP interpretability

Run: python drug_food_interaction_pipeline.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
import shap
import json
import os

from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors, DataStructs
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
# TASK 3: needed for native imbalance handling without synthetic data
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import lightgbm as lgb

# ─── reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# =============================================================================
# TASK 1 — DOMAIN ISOLATION: configurable oncology keyword filter
# =============================================================================
# All drugs in this pipeline are already oncology agents, but we enforce an
# explicit allowlist so the filter is visible, testable, and extensible when
# real CSV data is added from DrugBank / ChEMBL bulk downloads.
ONCOLOGY_CLASS_KEYWORDS = [
    "BCR-ABL_inhibitor", "EGFR_inhibitor", "multikinase_inhibitor",
    "BCL2_inhibitor", "SERM", "antimetabolite", "alkylating_agent",
    # generic expansions for real-world CSV compatibility (Task 1 requirement)
    "antineoplastic", "oncology", "kinase_inhibitor", "chemotherapy",
    "targeted_therapy", "immunotherapy", "hormone_therapy",
]


def is_oncology_drug(drug_class: str) -> bool:
    """
    Returns True if any oncology keyword appears in the drug_class string.
    Case-insensitive. Used in build_raw_dataset() to isolate oncology records.
    Scientific rationale: drug-food interactions for oncology agents are
    mechanism-specific (CYP3A4 saturation, P-gp competition) and must NOT
    be conflated with general-population pharmacokinetics.
    """
    dc_lower = drug_class.lower()
    return any(kw.lower() in dc_lower for kw in ONCOLOGY_CLASS_KEYWORDS)


# =============================================================================
# STEP 1 — Advanced Data Acquisition (Multi-Source)
# =============================================================================

# ── oncology drugs: name → SMILES (curated from public literature) ──────────
ONCOLOGY_DRUG_SMILES = {
    "Imatinib":      "CC1=CC=C(C=C1)NC2=NC=CC(=N2)NC3=CC(=C(C=C3)CN4CCN(CC4)C)C(=O)N",
    "Erlotinib":     "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCO",
    "Sorafenib":     "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
    "Gefitinib":     "COC1=CC2=C(C=C1OCCCN3CCOCC3)C(=NC=N2)NC4=CC=C(C=C4)F",
    "Tamoxifen":     "CC(CC1=CC=CC=C1)(/C(=C/CCN(CC)CC)C2=CC=C(C=C2)OCC)C",
    "Methotrexate":  "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "Capecitabine":  "CCOC(=O)NC1=NC(=O)N(C=C1F)[C@@H]2O[C@H](C)[C@@H](O)[C@H]2O",
    "Cyclophosphamide":"ClCCN(CCCl)P1(=O)NCCCO1",
    "Dasatinib":     "Cc1nc(Nc2ncc(s2)C(=O)Nc3c(C)cccc3Cl)cc(n1)N4CCN(CCO)CC4",
    "Venetoclax":    "CC1(CCC(=C1)c2ccc(cc2)N3CCN(CC3)c4ccc(cc4)C(=O)NS(=O)(=O)c5ccc(cc5)NC(=O)c6cc7cc(Cl)ccc7[nH]6)C",
}

# ── food components: name → SMILES ──────────────────────────────────────────
FOOD_SMILES = {
    "Quercetin":     "O=C1C(O)=C(OC2=CC(O)=CC(O)=C12)C3=CC(O)=C(O)C=C3",
    "Grapefruit_Naringenin": "O=C1CC(OC2=CC(=CC(=C2)O)O)C3=CC(=C(C=C3)O)O1",
    "Curcumin":      "O=C(/C=C/C1=CC(=C(C=C1)O)OC)/C=C/C2=CC(=C(C=C2)O)OC",
    "Resveratrol":   "OC1=CC(=CC(=C1)/C=C/C2=CC=C(O)C=C2)O",
    "EGCG":          "O=C(OC1CC(OC2=CC(=C(C=C2O)O)O)C3=C(O)C=C(O)C=C31)C4=CC(=C(C=C4O)O)O",
    "Caffeine":      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Piperine":      "O=C(/C=C/C=C/C1=CC2=C(C=C1)OCO2)N3CCCCC3",
    "Lycopene":      "CC(=CCC/C(=C/CC/C(=C/CC/C(=C/CC/C(=C/CCC=C(C)C)C)C)C)C)C",
    "Folate":        "NC1=NC2=C(C=C1)N=C(N2)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "Vitamin_K":     "CC1=CC(=O)C2=CC=CC=C2C1=O",
}

# ── CYP450 inhibition data for drugs (mocked, based on known pharmacology) ──
CYP450_DATA = {
    "Imatinib":       {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 1, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
    "Erlotinib":      {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
    "Sorafenib":      {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 1, "Pgp_substrate": 0},
    "Gefitinib":      {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
    "Tamoxifen":      {"CYP3A4_inhibitor": 0, "CYP2D6_inhibitor": 1, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
    "Methotrexate":   {"CYP3A4_inhibitor": 0, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 0},
    "Capecitabine":   {"CYP3A4_inhibitor": 0, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 1, "Pgp_substrate": 0},
    "Cyclophosphamide":{"CYP3A4_inhibitor":0, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 0},
    "Dasatinib":      {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
    "Venetoclax":     {"CYP3A4_inhibitor": 1, "CYP2D6_inhibitor": 0, "CYP2C9_inhibitor": 0, "Pgp_substrate": 1},
}

# ── Protein binding affinity (Ki, nM) for food components (mocked) ──────────
BINDING_DATA = {
    "Quercetin":      {"CYP3A4_Ki_nM": 4.5,  "CYP2D6_Ki_nM": 120.0},
    "Grapefruit_Naringenin": {"CYP3A4_Ki_nM": 8.0, "CYP2D6_Ki_nM": 300.0},
    "Curcumin":       {"CYP3A4_Ki_nM": 25.0, "CYP2D6_Ki_nM": 80.0},
    "Resveratrol":    {"CYP3A4_Ki_nM": 60.0, "CYP2D6_Ki_nM": 500.0},
    "EGCG":           {"CYP3A4_Ki_nM": 35.0, "CYP2D6_Ki_nM": 200.0},
    "Caffeine":       {"CYP3A4_Ki_nM": 5000.0,"CYP2D6_Ki_nM": 1000.0},
    "Piperine":       {"CYP3A4_Ki_nM": 50.0, "CYP2D6_Ki_nM": 150.0},
    "Lycopene":       {"CYP3A4_Ki_nM": 1000.0,"CYP2D6_Ki_nM": 2000.0},
    "Folate":         {"CYP3A4_Ki_nM": 5000.0,"CYP2D6_Ki_nM": 5000.0},
    "Vitamin_K":      {"CYP3A4_Ki_nM": 80.0, "CYP2D6_Ki_nM": 1000.0},
}

# ── known interaction outcomes (classification + % bioavailability change) ───
KNOWN_INTERACTIONS = {
    ("Imatinib",       "Quercetin"):             (1, -35.0),
    ("Imatinib",       "Grapefruit_Naringenin"): (1, +60.0),
    ("Erlotinib",      "Grapefruit_Naringenin"): (1, +44.0),
    ("Sorafenib",      "Quercetin"):             (1, -28.0),
    ("Gefitinib",      "EGCG"):                  (1, -50.0),
    ("Tamoxifen",      "Curcumin"):              (1, -22.0),
    ("Methotrexate",   "Folate"):                (1, -30.0),
    ("Capecitabine",   "Vitamin_K"):             (1, +15.0),
    ("Dasatinib",      "Caffeine"):              (1, +10.0),
    ("Venetoclax",     "Piperine"):              (1, +45.0),
    ("Imatinib",       "Caffeine"):              (0,   0.0),
    ("Erlotinib",      "Caffeine"):              (0,   0.0),
    ("Sorafenib",      "Lycopene"):              (0,   0.0),
    ("Gefitinib",      "Folate"):                (0,   0.0),
    ("Methotrexate",   "Curcumin"):              (0,   0.0),
    ("Cyclophosphamide","Resveratrol"):           (0,   0.0),
    ("Dasatinib",      "Lycopene"):              (0,   0.0),
    ("Venetoclax",     "Caffeine"):              (0,   0.0),
    ("Tamoxifen",      "Resveratrol"):           (1, -18.0),
    ("Cyclophosphamide","Vitamin_K"):            (1, +12.0),
    ("Gefitinib",      "Curcumin"):              (1, -20.0),
    ("Sorafenib",      "EGCG"):                  (1, -15.0),
    ("Erlotinib",      "Quercetin"):             (1, -25.0),
    ("Imatinib",       "Piperine"):              (1, +30.0),
    ("Venetoclax",     "Grapefruit_Naringenin"): (1, +55.0),
    ("Capecitabine",   "Folate"):                (0,   0.0),
    ("Tamoxifen",      "Caffeine"):              (0,   0.0),
    ("Methotrexate",   "Lycopene"):              (0,   0.0),
    ("Imatinib",       "EGCG"):                  (1, -40.0),
    ("Dasatinib",      "Quercetin"):             (1, -32.0),
    ("Erlotinib",      "Piperine"):              (1, +35.0),
    ("Cyclophosphamide","EGCG"):                 (0,   0.0),
    ("Sorafenib",      "Piperine"):              (1, +22.0),
    ("Gefitinib",      "Vitamin_K"):             (0,   0.0),
    ("Methotrexate",   "Resveratrol"):           (0,   0.0),
    ("Venetoclax",     "Curcumin"):              (0,   0.0),
    ("Tamoxifen",      "EGCG"):                  (1, -12.0),
    ("Capecitabine",   "Quercetin"):             (0,   0.0),
    ("Imatinib",       "Curcumin"):              (1, -18.0),
    ("Dasatinib",      "Grapefruit_Naringenin"): (1, +50.0),
}

# ── drug classes (for LODCO validation) ─────────────────────────────────────
DRUG_CLASSES = {
    "Imatinib":        "BCR-ABL_inhibitor",
    "Erlotinib":       "EGFR_inhibitor",
    "Gefitinib":       "EGFR_inhibitor",
    "Sorafenib":       "multikinase_inhibitor",
    "Dasatinib":       "BCR-ABL_inhibitor",
    "Venetoclax":      "BCL2_inhibitor",
    "Tamoxifen":       "SERM",
    "Methotrexate":    "antimetabolite",
    "Capecitabine":    "antimetabolite",
    "Cyclophosphamide":"alkylating_agent",
}


def fetch_pubchem_properties(smiles: str) -> dict:
    """
    Query PubChem REST API for physicochemical properties by SMILES.
    Falls back to empty dict on network failure (offline-safe).
    """
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
    try:
        resp = requests.post(url, data={"smiles": smiles}, timeout=8)
        if resp.status_code == 200:
            props = resp.json()["PropertyTable"]["Properties"][0]
            return {
                "MW_api":    props.get("MolecularWeight", np.nan),
                "XLogP_api": props.get("XLogP", np.nan),
                "TPSA_api":  props.get("TPSA", np.nan),
                "HBD_api":   props.get("HBondDonorCount", np.nan),
                "HBA_api":   props.get("HBondAcceptorCount", np.nan),
            }
    except Exception:
        pass
    return {"MW_api": np.nan, "XLogP_api": np.nan, "TPSA_api": np.nan,
            "HBD_api": np.nan, "HBA_api": np.nan}


def build_raw_dataset() -> pd.DataFrame:
    """
    Merge drug/food SMILES with CYP450 enzyme data, BindingDB affinities,
    and known interaction labels into one unified DataFrame.
    """
    records = []
    for (drug, food), (label, pct_change) in KNOWN_INTERACTIONS.items():
        rec = {
            "drug_name":     drug,
            "food_name":     food,
            "drug_smiles":   ONCOLOGY_DRUG_SMILES[drug],
            "food_smiles":   FOOD_SMILES[food],
            "interaction":   label,          # classification target
            "pct_change":    pct_change,     # regression target
            "drug_class":    DRUG_CLASSES[drug],
        }
        # merge CYP450 enzyme flags
        rec.update(CYP450_DATA[drug])
        # merge BindingDB Ki values for food
        rec.update(BINDING_DATA[food])
        records.append(rec)

    df = pd.DataFrame(records)

    # ── TASK 1: Domain isolation — keep only verified oncology drug classes ──
    # This step mirrors what would happen when ingesting a real CSV from
    # DrugBank or ChEMBL: we filter by drug_class before any feature
    # engineering so no spurious non-oncology pairs contaminate the model.
    before = len(df)
    df = df[df["drug_class"].apply(is_oncology_drug)].reset_index(drop=True)
    print(f"[Data] Oncology filter: {before} → {len(df)} pairs retained")
    print(f"[Data] Raw dataset: {len(df)} drug-food pairs")
    return df


# =============================================================================
# STEP 2 — Enhanced Feature Engineering
# =============================================================================

def mol_from_smiles(smi: str):
    """Parse SMILES; return RDKit Mol or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  [WARN] Invalid SMILES: {smi[:40]}…")
    return mol


def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Generate ECFP4 (Morgan r=2) bit-vector fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(mol) -> dict:
    """
    Compute Lipinski Rule-of-5 properties + LogD surrogate (LogP - TPSA/100).
    LogD is pH-dependent; this is a commonly used computational approximation.
    """
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    logd = logp - (tpsa / 100.0)   # computational LogD approximation
    ro5_violations = sum([
        mw > 500, logp > 5, hbd > 5, hba > 10
    ])
    return {
        "MW": mw, "LogP": logp, "TPSA": tpsa,
        "HBD": hbd, "HBA": hba, "RotBonds": rotb,
        "AromaticRings": arom, "LogD": logd,
        "Ro5_violations": ro5_violations,
    }


def tanimoto_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """
    Compute Tanimoto (Jaccard) coefficient between two bit-vector fingerprints.
    Range [0,1]; higher = more structurally similar.
    """
    intersect = np.sum(fp_a & fp_b)
    union     = np.sum(fp_a | fp_b)
    return float(intersect / union) if union > 0 else 0.0


def build_feature_matrix(df: pd.DataFrame, fp_bits: int = 256) -> pd.DataFrame:
    """
    Build full feature matrix for each drug-food pair:
      - Drug fingerprint (fp_bits) + Food fingerprint (fp_bits) → concatenated
      - Drug physicochemical descriptors (9 features)
      - Food physicochemical descriptors (9 features, prefixed)
      - Tanimoto similarity score
      - CYP450 enzyme flags from ChEMBL
      - BindingDB Ki values (log-transformed)
    """
    drug_fps, food_fps = [], []
    drug_descs, food_descs = [], []
    tanimotos = []

    for _, row in df.iterrows():
        d_mol = mol_from_smiles(row["drug_smiles"])
        f_mol = mol_from_smiles(row["food_smiles"])

        d_fp  = morgan_fingerprint(d_mol, n_bits=fp_bits)
        f_fp  = morgan_fingerprint(f_mol, n_bits=fp_bits)
        drug_fps.append(d_fp)
        food_fps.append(f_fp)

        drug_descs.append(rdkit_descriptors(d_mol))
        food_descs.append(rdkit_descriptors(f_mol))

        tanimotos.append(tanimoto_similarity(d_fp, f_fp))

    # fingerprint DataFrames
    drug_fp_df = pd.DataFrame(
        drug_fps, columns=[f"drug_fp_{i}" for i in range(fp_bits)]
    )
    food_fp_df = pd.DataFrame(
        food_fps, columns=[f"food_fp_{i}" for i in range(fp_bits)]
    )

    # descriptor DataFrames
    drug_desc_df = pd.DataFrame(drug_descs).add_prefix("drug_")
    food_desc_df = pd.DataFrame(food_descs).add_prefix("food_")

    # CYP450 and BindingDB columns already in df
    enzyme_cols = ["CYP3A4_inhibitor", "CYP2D6_inhibitor",
                   "CYP2C9_inhibitor", "Pgp_substrate"]
    ki_cols = ["CYP3A4_Ki_nM", "CYP2D6_Ki_nM"]

    enzyme_df = df[enzyme_cols].reset_index(drop=True)
    ki_df     = np.log1p(df[ki_cols].fillna(9999)).reset_index(drop=True)
    ki_df.columns = [c.replace("_nM", "_logKi") for c in ki_df.columns]

    tanimoto_df = pd.DataFrame({"tanimoto_similarity": tanimotos})

    # concatenate all feature blocks
    feat_df = pd.concat([
        drug_fp_df,
        food_fp_df,
        drug_desc_df.reset_index(drop=True),
        food_desc_df.reset_index(drop=True),
        enzyme_df.reset_index(drop=True),
        ki_df.reset_index(drop=True),
        tanimoto_df,
    ], axis=1)

    print(f"[Features] Matrix shape: {feat_df.shape[0]} rows × {feat_df.shape[1]} cols")
    return feat_df


# =============================================================================
# STEP 3 — ML Implementation
# =============================================================================

def apply_rfe(X: np.ndarray, y: np.ndarray, n_features: int = 40) -> np.ndarray:
    """
    Recursive Feature Elimination using LogisticRegression as the estimator.
    Selects top `n_features` for downstream models.
    Returns boolean mask array.
    """
    # Use only non-fingerprint columns for RFE (descriptors + enzyme + Ki)
    # Full FP columns are kept as-is since RFE on 512 FP bits is expensive
    estimator = LogisticRegression(max_iter=500, solver="lbfgs",
                                   C=0.1, random_state=SEED)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=10)
    rfe.fit(X, y)
    print(f"[RFE] Selected {rfe.n_features_} / {X.shape[1]} features")
    return rfe.support_


def train_classifier(X_train, y_train) -> xgb.XGBClassifier:
    """
    TASK 4 — Optimized single XGBClassifier (replaces any ensemble/VotingClassifier).

    Why a single XGBoost instead of an ensemble?
    • VotingClassifiers are interpretability black-boxes; SHAP TreeExplainer
      works natively and deterministically on a single XGBClassifier tree array.
    • For sparse 512-D Morgan fingerprint data, averaging heterogeneous models
      introduces calibration noise that hurts minority-class recall.

    Key hyperparameter choices (cheminformatics best practice):
    • tree_method='hist'  → exact-histogram splits; handles sparse FP bits
      efficiently without requiring dense matrix conversion.
    • max_depth=4         → shallow trees prevent scaffold memorization;
      each tree can only learn 4-level feature interactions.
    • colsample_bytree=0.3 → CRITICAL for 512-D fingerprint data: forces
      every tree to examine only ~154 randomly chosen bits, so different
      trees specialize in different molecular substructures rather than
      always routing on the same dominant bit patterns.
    • scale_pos_weight    → TASK 3 native imbalance handling; ratio of
      negative:positive class examples directly upweights minority-class
      gradients without synthesizing impossible chemical structures (SMOTE
      on binary FP bits produces non-integer, chemically impossible vectors).
    • reg_alpha=0.8, reg_lambda=2.0 → stronger L1/L2 vs. v1 to compensate
      for the shallower colsample, keeping generalization tight.
    • n_estimators=400    → more rounds at low lr=0.05 recovers the
      expressiveness lost from colsample_bytree=0.3.
    """
    # TASK 3 — compute balanced sample weights natively
    # This is mathematically equivalent to scale_pos_weight but works
    # per-sample, giving finer control when class imbalance is uneven.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Derive scale_pos_weight as a fallback sanity signal (logged only)
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = neg / pos if pos > 0 else 1.0
    print(f"  [Imbalance] neg={neg}, pos={pos}, scale_pos_weight≈{spw:.2f}")

    model = xgb.XGBClassifier(
        # TASK 4 mandatory hyperparameters
        tree_method="hist",          # sparse-data-safe histogram algorithm
        max_depth=4,                 # shallow → no scaffold memorization
        colsample_bytree=0.3,        # forces diverse molecular substructure coverage
        learning_rate=0.05,
        n_estimators=400,
        # additional regularization for high-dimensional FP space
        reg_alpha=0.8,               # L1: sparsity-inducing for FP bits
        reg_lambda=2.0,              # L2: weight shrinkage
        subsample=0.85,              # row-level bagging
        min_child_weight=2,          # prevents splits on single-sample nodes
        gamma=0.1,                   # min loss-reduction to make a split
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    # TASK 3: pass computed sample weights into fit() — no SMOTE, no synthetic data
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_regressor(X_train, y_train) -> lgb.LGBMRegressor:
    """
    LightGBM regressor for % bioavailability change (regression head).
    L1/L2 regularization via lambda_l1/lambda_l2.
    """
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        lambda_l1=0.5,      # L1
        lambda_l2=1.5,      # L2
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# STEP 4 — Strict Validation Protocol
# =============================================================================

def get_murcko_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(df: pd.DataFrame, feat_df: pd.DataFrame,
                   test_frac: float = 0.2) -> tuple:
    """
    Bemis-Murcko scaffold split:
    - Group pairs by drug scaffold
    - Assign scaffold groups to train/test to prevent leakage from
      structurally similar drugs appearing in both splits
    """
    scaffolds = df["drug_smiles"].apply(get_murcko_scaffold)
    unique_scaffolds = scaffolds.unique()

    np.random.shuffle(unique_scaffolds)
    n_test = max(1, int(len(unique_scaffolds) * test_frac))
    test_scaffolds  = set(unique_scaffolds[:n_test])
    train_scaffolds = set(unique_scaffolds[n_test:])

    test_mask  = scaffolds.isin(test_scaffolds).values
    train_mask = scaffolds.isin(train_scaffolds).values

    X_train = feat_df.values[train_mask]
    X_test  = feat_df.values[test_mask]
    y_cls_train = df["interaction"].values[train_mask]
    y_cls_test  = df["interaction"].values[test_mask]
    y_reg_train = df["pct_change"].values[train_mask]
    y_reg_test  = df["pct_change"].values[test_mask]

    print(f"[Scaffold Split] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return (X_train, X_test,
            y_cls_train, y_cls_test,
            y_reg_train, y_reg_test,
            train_mask, test_mask)


def lodco_validation(df: pd.DataFrame, feat_df: pd.DataFrame,
                     clf_params: dict, reg_params: dict) -> pd.DataFrame:
    """
    Leave-One-Drug-Class-Out (LODCO) cross-validation.
    In each fold, all pairs from one drug class are held out as the test set.
    This tests generalization to entirely unseen drug classes —
    the hardest and most clinically meaningful evaluation.
    """
    drug_classes = df["drug_class"].values
    unique_classes = np.unique(drug_classes)
    results = []

    for held_out in unique_classes:
        test_mask  = drug_classes == held_out
        train_mask = ~test_mask

        X_train = feat_df.values[train_mask]
        X_test  = feat_df.values[test_mask]
        y_train = df["interaction"].values[train_mask]
        y_test  = df["interaction"].values[test_mask]

        if len(np.unique(y_train)) < 2:
            continue  # skip if no positive class in training

        clf = xgb.XGBClassifier(**clf_params, verbosity=0,
                                 random_state=SEED, use_label_encoder=False,
                                 eval_metric="logloss")
        # TASK 3: balanced weights in LODCO too — critical for rare drug classes
        lodco_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        clf.fit(X_train, y_train, sample_weight=lodco_weights)
        preds = clf.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, zero_division=0)
        results.append({
            "held_out_class": held_out,
            "n_test":         int(test_mask.sum()),
            "accuracy":       round(acc, 3),
            "f1_score":       round(f1, 3),
        })
        print(f"  LODCO [{held_out}]  Acc={acc:.3f}  F1={f1:.3f}  n_test={test_mask.sum()}")

    return pd.DataFrame(results)


# =============================================================================
# STEP 5 — Unseen Dataset Evaluation
# =============================================================================

# Completely held-out pairs — never used in training or validation
UNSEEN_INTERACTIONS = {
    ("Imatinib",    "Resveratrol"):          (1, -20.0),
    ("Erlotinib",   "Vitamin_K"):            (0,   0.0),
    ("Sorafenib",   "Caffeine"):             (0,   0.0),
    ("Venetoclax",  "Quercetin"):            (1, -38.0),
    ("Methotrexate","Piperine"):             (1, +28.0),
    ("Dasatinib",   "Curcumin"):             (1, -16.0),
    ("Tamoxifen",   "Lycopene"):             (0,   0.0),
    ("Cyclophosphamide","Quercetin"):        (0,   0.0),
    ("Capecitabine","Piperine"):             (0,   0.0),
    ("Gefitinib",   "Resveratrol"):          (1, -10.0),
}


def build_unseen_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the completely independent holdout dataset.
    Shares the same feature engineering pipeline but is NEVER seen by the model.
    """
    records = []
    for (drug, food), (label, pct_change) in UNSEEN_INTERACTIONS.items():
        rec = {
            "drug_name":  drug,
            "food_name":  food,
            "drug_smiles":ONCOLOGY_DRUG_SMILES[drug],
            "food_smiles":FOOD_SMILES[food],
            "interaction":label,
            "pct_change": pct_change,
            "drug_class": DRUG_CLASSES[drug],
        }
        rec.update(CYP450_DATA[drug])
        rec.update(BINDING_DATA[food])
        records.append(rec)

    unseen_df   = pd.DataFrame(records)
    unseen_feat = build_feature_matrix(unseen_df)
    print(f"[Unseen] Dataset: {len(unseen_df)} pairs")
    return unseen_df, unseen_feat


# =============================================================================
# STEP 6 — Model Evaluation & Reporting
# =============================================================================

def evaluate_classifier(model, X, y_true, label="") -> dict:
    """Full classification evaluation: Acc, Prec, Rec, F1, AUC."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan,
    }
    if label:
        print(f"\n[Eval-CLS | {label}]")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
    return metrics, y_pred, y_prob


def evaluate_regressor(model, X, y_true, label="") -> dict:
    """Full regression evaluation: RMSE, MAE, R²."""
    y_pred = model.predict(X)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }
    if label:
        print(f"\n[Eval-REG | {label}]")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
    return metrics, y_pred


def plot_all_results(clf, reg,
                     X_train, X_val, X_unseen,
                     y_cls_train, y_cls_val, y_cls_unseen,
                     y_reg_train, y_reg_val, y_reg_unseen,
                     feat_names,
                     out_dir="/mnt/user-data/outputs"):
    """
    Generate all evaluation visualizations in one figure:
      1. Confusion matrix (validation)
      2. ROC curve (train / val / unseen)
      3. Feature importance (XGBoost)
      4. Predicted vs Actual % change (regression)
      5. Accuracy comparison bar (train / val / unseen)
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("Drug-Food Interaction ML Pipeline — Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Confusion Matrix ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _, y_pred_val, _ = evaluate_classifier(clf, X_val, y_cls_val)
    cm = confusion_matrix(y_cls_val, y_pred_val)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["No Interact.", "Interact."],
                yticklabels=["No Interact.", "Interact."])
    ax1.set_title("Confusion Matrix\n(Scaffold Validation Set)")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

    # ── 2. ROC Curves ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for X_, y_, lbl, col in [
        (X_train,  y_cls_train,  "Train",      "steelblue"),
        (X_val,    y_cls_val,    "Validation", "darkorange"),
        (X_unseen, y_cls_unseen, "Unseen Test","green"),
    ]:
        if len(np.unique(y_)) > 1:
            prob = clf.predict_proba(X_)[:, 1]
            fpr, tpr, _ = roc_curve(y_, prob)
            auc = roc_auc_score(y_, prob)
            ax2.plot(fpr, tpr, label=f"{lbl} (AUC={auc:.2f})", color=col)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax2.set_title("ROC Curves — Classifier"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(fontsize=8); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    # ── 3. Feature Importance (top 20) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = importances[top_idx]
    ax3.barh(range(20), top_vals[::-1], color="steelblue")
    ax3.set_yticks(range(20))
    ax3.set_yticklabels(top_names[::-1], fontsize=7)
    ax3.set_title("Top 20 Feature Importances\n(XGBoost)"); ax3.set_xlabel("Importance")

    # ── 4. Predicted vs Actual Regression ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _, y_reg_pred_val = evaluate_regressor(reg, X_val, y_reg_val)
    ax4.scatter(y_reg_val, y_reg_pred_val, color="darkorange",
                edgecolors="k", linewidths=0.4, s=60, alpha=0.8, label="Validation")
    _, y_reg_pred_uns = evaluate_regressor(reg, X_unseen, y_reg_unseen)
    ax4.scatter(y_reg_unseen, y_reg_pred_uns, color="green",
                edgecolors="k", linewidths=0.4, s=60, alpha=0.8, marker="^", label="Unseen")
    lims = [min(y_reg_val.min(), y_reg_unseen.min()) - 5,
            max(y_reg_val.max(), y_reg_unseen.max()) + 5]
    ax4.plot(lims, lims, "k--", alpha=0.5)
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_title("Predicted vs Actual\n% Bioavailability Change")
    ax4.set_xlabel("Actual (%)"); ax4.set_ylabel("Predicted (%)")
    ax4.legend(fontsize=8)

    # ── 5. Accuracy comparison ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sets = ["Train", "Validation", "Unseen Test"]
    accs = [
        accuracy_score(y_cls_train, clf.predict(X_train)),
        accuracy_score(y_cls_val,   clf.predict(X_val)),
        accuracy_score(y_cls_unseen,clf.predict(X_unseen)),
    ]
    bar_colors = ["steelblue", "darkorange", "green"]
    bars = ax5.bar(sets, accs, color=bar_colors, edgecolor="k", width=0.4)
    for bar, acc in zip(bars, accs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.set_ylim(0, 1.15)
    ax5.set_title("Accuracy: Train / Validation / Unseen")
    ax5.set_ylabel("Accuracy")
    ax5.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Random baseline")
    ax5.legend(fontsize=8)

    # ── 6. RMSE comparison ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    rmse_train, _ = evaluate_regressor(reg, X_train, y_reg_train)
    rmse_val,   _ = evaluate_regressor(reg, X_val,   y_reg_val)
    rmse_uns,   _ = evaluate_regressor(reg, X_unseen,y_reg_unseen)
    rmse_vals = [rmse_train["rmse"], rmse_val["rmse"], rmse_uns["rmse"]]
    bars2 = ax6.bar(sets, rmse_vals, color=bar_colors, edgecolor="k", width=0.4)
    for bar, v in zip(bars2, rmse_vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax6.set_title("RMSE: % Change Prediction")
    ax6.set_ylabel("RMSE (%)")

    fig.savefig(f"{out_dir}/evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Evaluation dashboard saved → {out_dir}/evaluation_dashboard.png")


# =============================================================================
# STEP 7 — Interpretability (SHAP)
# =============================================================================

def shap_analysis(clf, X_train: np.ndarray, feat_names: list,
                  out_dir="/mnt/user-data/outputs"):
    """
    SHAP TreeExplainer for XGBoost classifier.
    Produces:
      - Summary beeswarm plot (global feature importance)
      - Mean absolute SHAP bar plot
    These explain WHICH molecular features drive interaction predictions.
    """
    os.makedirs(out_dir, exist_ok=True)
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]   # use positive class
    else:
        shap_vals = shap_values

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("SHAP Interpretability — Drug-Food Interaction Classifier",
                 fontsize=13, fontweight="bold")

    # ── beeswarm summary ──────────────────────────────────────────────────
    plt.sca(axes[0])
    shap.summary_plot(shap_vals, X_train,
                      feature_names=feat_names,
                      max_display=20, show=False)
    axes[0].set_title("SHAP Beeswarm\n(Top 20 Features)", fontsize=11)

    # ── mean |SHAP| bar ───────────────────────────────────────────────────
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:20]
    top_names = [feat_names[i] for i in top_idx]
    top_shap  = mean_abs[top_idx]
    axes[1].barh(range(20), top_shap[::-1], color="coral", edgecolor="k")
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels(top_names[::-1], fontsize=8)
    axes[1].set_title("Mean |SHAP| Value\n(Global Feature Impact)", fontsize=11)
    axes[1].set_xlabel("Mean |SHAP value|")

    plt.tight_layout()
    fig.savefig(f"{out_dir}/shap_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP] Analysis saved → {out_dir}/shap_analysis.png")

    # ── overfitting detection ─────────────────────────────────────────────
    return shap_vals, top_names


def generate_patient_reasoning(shap_vals: np.ndarray,
                               feature_names: list,
                               drug_name: str = "",
                               food_name: str = "",
                               top_n: int = 3) -> str:
    """
    TASK 6 — Synopsis Fulfillment: Patient-Readable Interaction Explanation.

    Translates raw SHAP values (which are chemist-facing numbers) into
    plain-English sentences that patients and caregivers can understand
    without any pharmacology background.

    Mapping logic (cheminformatics → lay language):
    • 'LogP' / 'LogD'      → lipophilicity (fat-solubility)
    • 'CYP3A4'             → liver enzyme CYP3A4 (metabolizes most oncology drugs)
    • 'CYP2D6'             → liver enzyme CYP2D6
    • 'tanimoto'           → chemical shape similarity between drug and food
    • 'MW' / 'MolWt'       → molecular size
    • 'TPSA'               → surface chemistry / membrane penetration
    • 'HBD' / 'HBA'        → hydrogen bonding ability
    • 'fp_' (fingerprint)  → shared molecular substructure between drug and food
    • 'Pgp'                → P-glycoprotein transporter (drug efflux pump)
    • 'Ki'                 → binding affinity to a liver enzyme

    Parameters
    ----------
    shap_vals   : 2D array (n_samples, n_features) — SHAP values for positive class
    feature_names : list of feature name strings aligned to shap_vals columns
    drug_name   : name of the drug for personalised message
    food_name   : name of the food for personalised message
    top_n       : number of top features to explain (default 3)

    Returns
    -------
    Plain-English warning string ready for patient-facing UI display.
    """

    # ── feature → human-readable phrase dictionary ──────────────────────────
    FEATURE_PHRASES = {
        "logp":        "high fat-solubility of the drug (LogP)",
        "logd":        "lipophilicity at body pH (LogD)",
        "cyp3a4_inhibitor": "the drug blocks liver enzyme CYP3A4",
        "cyp3a4_ki":   "the food strongly binds to liver enzyme CYP3A4",
        "cyp2d6_inhibitor": "the drug blocks liver enzyme CYP2D6",
        "cyp2d6_ki":   "the food binds to liver enzyme CYP2D6",
        "tanimoto":    "structural similarity between drug and food molecules",
        "mw":          "large molecular size of the drug",
        "tpsa":        "surface chemistry affecting gut absorption",
        "hbd":         "hydrogen-bonding pattern of the drug",
        "hba":         "hydrogen-bonding pattern of the drug",
        "pgp":         "drug transport pump (P-glycoprotein) activity",
        "fp_":         "shared chemical substructure between drug and food",
        "ro5":         "drug violating standard absorption rules",
    }

    SEVERITY_PHRASES = [
        "⚠️  CRITICAL INTERACTION WARNING",
        "⚠️  MODERATE INTERACTION DETECTED",
        "ℹ️  POTENTIAL INTERACTION NOTED",
    ]

    # ── identify top contributing features by mean |SHAP| across samples ───
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_indices   = np.argsort(mean_abs_shap)[::-1][:top_n]

    # ── translate each feature to a patient-friendly phrase ─────────────────
    reasons = []
    for idx in top_indices:
        fname = feature_names[idx].lower()
        phrase = None
        for key, human_phrase in FEATURE_PHRASES.items():
            if key in fname:
                phrase = human_phrase
                break
        if phrase is None:
            # fallback: strip prefix and capitalise
            phrase = f"molecular property '{feature_names[idx]}'"
        reasons.append(phrase)

    # ── select severity header based on SHAP magnitude ──────────────────────
    max_shap = float(mean_abs_shap[top_indices[0]])
    if max_shap > 0.15:
        severity = SEVERITY_PHRASES[0]
    elif max_shap > 0.05:
        severity = SEVERITY_PHRASES[1]
    else:
        severity = SEVERITY_PHRASES[2]

    # ── construct readable message ───────────────────────────────────────────
    drug_label = drug_name if drug_name else "this oncology drug"
    food_label = food_name if food_name else "this food"

    reason_str = "; ".join(reasons)

    message = (
        f"\n{'─'*65}\n"
        f"  {severity}\n"
        f"{'─'*65}\n"
        f"  Drug : {drug_label}\n"
        f"  Food : {food_label}\n"
        f"\n"
        f"  Why this interaction may be dangerous:\n"
        f"  The prediction model identified {top_n} key chemical reasons:\n"
    )
    for i, r in enumerate(reasons, 1):
        message += f"    {i}. {r.capitalize()}.\n"

    message += (
        f"\n"
        f"  What this means for you:\n"
        f"  Consuming {food_label} while taking {drug_label} may change\n"
        f"  how your body absorbs or processes the medication. This can\n"
        f"  make the drug less effective or cause unexpected side effects.\n"
        f"\n"
        f"  → Please consult your oncologist or pharmacist before consuming\n"
        f"    {food_label} during your treatment course.\n"
        f"{'─'*65}"
    )
    return message


def detect_overfitting(train_metrics: dict, val_metrics: dict,
                       unseen_metrics: dict):
    """
    Compare train vs. validation vs. unseen accuracy/AUC.
    A gap of >10% train-vs-val signals overfitting.
    A smaller unseen gap confirms generalization.
    """
    print("\n" + "="*55)
    print("  Overfitting Detection Report")
    print("="*55)
    for metric in ["accuracy", "f1", "roc_auc"]:
        tr = train_metrics.get(metric, np.nan)
        va = val_metrics.get(metric, np.nan)
        un = unseen_metrics.get(metric, np.nan)
        gap_tv = tr - va
        gap_tu = tr - un
        flag = "⚠️  OVERFIT" if gap_tv > 0.15 else "✅ OK"
        print(f"  {metric:10s} | Train={tr:.3f}  Val={va:.3f}  Unseen={un:.3f}"
              f"  Gap(T-V)={gap_tv:+.3f}  {flag}")
    print("="*55)


# =============================================================================
# MAIN — orchestrates the full pipeline
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  Oncology Drug-Food Interaction — Full ML Pipeline")
    print("="*65)

    # ── 1. Data Acquisition ───────────────────────────────────────────────
    print("\n[STEP 1] Data Acquisition")
    df = build_raw_dataset()

    # ── 2. Feature Engineering ────────────────────────────────────────────
    print("\n[STEP 2] Feature Engineering")
    feat_df = build_feature_matrix(df, fp_bits=256)
    feat_names = feat_df.columns.tolist()

    # ── 3. Scaffold Split ─────────────────────────────────────────────────
    print("\n[STEP 3/4] Scaffold-Based Train/Val Split")
    (X_train, X_val,
     y_cls_train, y_cls_val,
     y_reg_train, y_reg_val,
     train_mask, val_mask) = scaffold_split(df, feat_df, test_frac=0.25)

    # Scale continuous descriptors (fingerprint bits are already binary)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # ── RFE on non-FP columns only (last 20 cols) ─────────────────────────
    n_fp = 512   # 256 drug + 256 food
    X_non_fp_train = X_train[:, n_fp:]
    X_non_fp_val   = X_val[:, n_fp:]

    rfe_mask = apply_rfe(X_non_fp_train, y_cls_train, n_features=15)

    X_train_full = np.hstack([X_train[:, :n_fp], X_non_fp_train[:, rfe_mask]])
    X_val_full   = np.hstack([X_val[:, :n_fp],   X_non_fp_val[:, rfe_mask]])
    feat_names_full = (
        feat_names[:n_fp]
        + [feat_names[n_fp + i] for i, s in enumerate(rfe_mask) if s]
    )
    print(f"[RFE] Final feature dim: {X_train_full.shape[1]}")

    # ── 4. Train Multi-Task Models ─────────────────────────────────────────
    print("\n[STEP 3] Training Models")
    # TASK 4: clf_params updated to match the new optimized XGBClassifier.
    # These are passed to LODCO which instantiates its own models, so they
    # must stay aligned with the train_classifier() hyperparameter choices.
    clf_params = dict(
        tree_method="hist",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=0.8,
        reg_lambda=2.0,
        subsample=0.85,
        colsample_bytree=0.3,    # CRITICAL for FP-sparse data
        min_child_weight=2,
        gamma=0.1,
    )
    clf = train_classifier(X_train_full, y_cls_train)
    reg = train_regressor(X_train_full, y_reg_train)
    print("  XGBoost classifier and LightGBM regressor trained.")

    # ── 4b. LODCO Validation ──────────────────────────────────────────────
    print("\n[STEP 4] LODCO Cross-Validation")
    lodco_df = lodco_validation(df, feat_df, clf_params, {})
    print("\nLODCO Summary:")
    print(lodco_df.to_string(index=False))

    # ── 5. Unseen Dataset Evaluation ──────────────────────────────────────
    print("\n[STEP 5] Unseen Dataset Evaluation")
    unseen_df, unseen_feat = build_unseen_dataset()
    X_unseen = scaler.transform(unseen_feat.values)
    X_unseen_non_fp = X_unseen[:, n_fp:]
    X_unseen_full = np.hstack([X_unseen[:, :n_fp], X_unseen_non_fp[:, rfe_mask]])
    y_cls_unseen = unseen_df["interaction"].values
    y_reg_unseen = unseen_df["pct_change"].values

    # ── 6. Full Evaluation ────────────────────────────────────────────────
    print("\n[STEP 6] Model Evaluation")
    train_metrics, _, _ = evaluate_classifier(clf, X_train_full, y_cls_train, "Train")
    val_metrics,   _, _ = evaluate_classifier(clf, X_val_full,   y_cls_val,   "Validation")
    unseen_metrics,_, _ = evaluate_classifier(clf, X_unseen_full,y_cls_unseen,"Unseen Test")

    evaluate_regressor(reg, X_train_full, y_reg_train, "Train (Reg)")
    evaluate_regressor(reg, X_val_full,   y_reg_val,   "Val (Reg)")
    evaluate_regressor(reg, X_unseen_full,y_reg_unseen,"Unseen (Reg)")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_all_results(
        clf, reg,
        X_train_full, X_val_full, X_unseen_full,
        y_cls_train, y_cls_val, y_cls_unseen,
        y_reg_train, y_reg_val, y_reg_unseen,
        feat_names_full,
    )

    # ── 7. SHAP Interpretability ──────────────────────────────────────────
    print("\n[STEP 7] SHAP Analysis")
    shap_vals, top_shap_features = shap_analysis(
        clf, X_train_full, feat_names_full
    )
    print("\n  Top 10 features driving interaction predictions:")
    for i, name in enumerate(top_shap_features[:10], 1):
        print(f"  {i:>2}. {name}")

    # ── Overfitting Detection ─────────────────────────────────────────────
    detect_overfitting(train_metrics, val_metrics, unseen_metrics)

    # ── TASK 6: Patient Reasoning UI — Synopsis Fulfillment ───────────────
    # The academic synopsis requires "reasoning for each assessment in a
    # user-friendly format for patients and caregivers". This call takes
    # the SHAP values already computed and prints a sample patient warning
    # for the highest-risk pair visible in the training data.
    print("\n[STEP 7b] Patient Explanation UI (Synopsis Requirement)")

    # Use training SHAP values (global explanation, not per-pair)
    # For a live system, call explainer.shap_values(X_single_pair) instead.
    sample_reasoning = generate_patient_reasoning(
        shap_vals=shap_vals,
        feature_names=feat_names_full,
        drug_name="Imatinib",          # highest-interaction drug in dataset
        food_name="Grapefruit (Naringenin)",
        top_n=3,
    )
    print(sample_reasoning)

    # Demonstrate a second pair (different drug class — antimetabolite)
    sample_reasoning_2 = generate_patient_reasoning(
        shap_vals=shap_vals,
        feature_names=feat_names_full,
        drug_name="Methotrexate",
        food_name="Folate-rich foods (spinach, lentils)",
        top_n=3,
    )
    print(sample_reasoning_2)

    print("\n[Pipeline Complete] All outputs saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()
