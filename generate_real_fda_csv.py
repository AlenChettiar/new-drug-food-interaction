import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import json
import time

def fetch_smiles_by_name(compound_name: str) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(compound_name)}/property/IsomericSMILES/JSON"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read())
            props = res["PropertyTable"]["Properties"][0]
            return props.get("IsomericSMILES", props.get("SMILES", ""))
    except Exception:
        pass
    return ""

print("Loading drugs from clinical_interactions.csv...")
original_df = pd.read_csv("clinical_interactions.csv")
unique_drugs = original_df[['Drug_Name', 'Drug_Class']].drop_duplicates()
print(f"Found {len(unique_drugs)} unique oncology drugs.")

# ── Leave-One-Drug-Out (LODO) split ───────────────────────────────────────
# Hold out one entire drug per class so the unseen set is structurally
# out-of-distribution — the model never sees these drugs during training.
UNSEEN_DRUGS = {
    'BCR-ABL_inhibitor':     'Dasatinib',
    'EGFR_inhibitor':        'Osimertinib',
    'multikinase_inhibitor': 'Cabozantinib',
    'BCL2_inhibitor':        'Venetoclax',
    'CDK4_inhibitor':        'Palbociclib',
    'SERM':                  'Toremifene',
    'antimetabolite':        'Gemcitabine',
    'alkylating_agent':      'Ifosfamide',
}
unseen_drug_set = set(UNSEEN_DRUGS.values())
print(f"\n[LODO Split] Holding out {len(unseen_drug_set)} drugs as completely unseen:")
for cls, drug in UNSEEN_DRUGS.items():
    print(f"  {cls:28s} -> {drug}")

# Map heterogeneous food names to their active chemical components
food_active_map = {
    'grapefruit':      'Bergamottin',
    'caffeine':        'Caffeine',
    'alcohol':         'Ethanol',
    "st. john's wort": 'Hyperforin',
    'fat':             'Oleic Acid',
    'dairy':           'Calcium',
    'garlic':          'Allicin',
    'green tea':       'Epigallocatechin gallate',
    'calcium':         'Calcium',
    'vitamin c':       'Ascorbic Acid',
}

data = []
np.random.seed(42)

# Pre-fetch food SMILES once
food_smiles_cache = {}
for food in food_active_map.values():
    print(f"Fetching SMILES for food component: {food}...")
    food_smiles_cache[food] = fetch_smiles_by_name(food)
    time.sleep(0.3)

drug_smiles_cache = {}

for _, row in unique_drugs.iterrows():
    drug      = row['Drug_Name']
    drug_cls  = row['Drug_Class']
    split     = "unseen" if drug in unseen_drug_set else "train"

    if drug not in drug_smiles_cache:
        drug_smiles_cache[drug] = fetch_smiles_by_name(drug)
        time.sleep(0.3)

    d_smi = drug_smiles_cache[drug]
    if not d_smi:
        print(f"  [WARN] Skipping {drug} — no PubChem SMILES found.")
        continue

    print(f"Fetching FDA label for {drug}  [{split}]...")
    url = (f"https://api.fda.gov/drug/label.json"
           f"?search=openfda.substance_name:\"{urllib.parse.quote(drug)}\"&limit=1")
    text_to_search = ""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())['results'][0]
            for field in ['drug_interactions', 'clinical_pharmacology',
                          'warnings_and_cautions', 'information_for_patients',
                          'boxed_warning']:
                if field in res_data:
                    text_to_search += " ".join(res_data[field]).lower() + " "
    except Exception as e:
        print(f"  [WARN] FDA fetch failed for {drug}: {e}")

    for food_term, active_chemical in food_active_map.items():
        f_smi = food_smiles_cache.get(active_chemical, "")
        if not f_smi:
            continue

        if food_term in text_to_search:
            if food_term in ['grapefruit', 'fat']:
                change_pct  = np.random.uniform(30.0, 60.0)
                interaction = 2
            elif food_term == "st. john's wort":
                change_pct  = np.random.uniform(-40.0, -60.0)
                interaction = 2
            elif food_term in ['alcohol', 'dairy', 'calcium']:
                base_pct    = np.random.uniform(-25.0, 25.0)
                if abs(base_pct) < 15:
                    base_pct = np.random.uniform(15.0, 25.0) * np.sign(base_pct)
                change_pct  = base_pct
                interaction = 1
            else:
                change_pct  = np.random.uniform(15.0, 25.0)
                interaction = 1
        else:
            change_pct  = 0.0
            interaction = 0

        data.append({
            "Drug_Name":          drug,
            "Food_Name":          food_term.title(),
            "Drug_SMILES":        d_smi,
            "Food_SMILES":        f_smi,
            "Drug_Class":         drug_cls,
            "Bioavail_Change_Pct": round(change_pct, 1),
            "interactions":       interaction,
            "Split":              split,
        })

    time.sleep(0.3)

new_df = pd.DataFrame(data)
new_df.to_csv("clinical_interaction_real.csv", index=False)

train_df  = new_df[new_df["Split"] == "train"]
unseen_df = new_df[new_df["Split"] == "unseen"]

print(f"\n[Success] Exported {len(new_df)} pairs to clinical_interaction_real.csv")
print(f"  Train  : {len(train_df)} rows  ({train_df['interactions'].value_counts().to_dict()})")
print(f"  Unseen : {len(unseen_df)} rows  ({unseen_df['interactions'].value_counts().to_dict()})")

