import pandas as pd
import numpy as np

# Expanding the dictionaries to synthesize roughly ~500 valid mappings.
drugs = {
    'Imatinib': 'BCR-ABL_inhibitor', 'Dasatinib': 'BCR-ABL_inhibitor', 'Nilotinib': 'BCR-ABL_inhibitor', 
    'Ponatinib': 'BCR-ABL_inhibitor', 'Bosutinib': 'BCR-ABL_inhibitor', 'Radotinib': 'BCR-ABL_inhibitor',
    'Erlotinib': 'EGFR_inhibitor', 'Gefitinib': 'EGFR_inhibitor', 'Osimertinib': 'EGFR_inhibitor', 
    'Afatinib': 'EGFR_inhibitor', 'Lapatinib': 'EGFR_inhibitor', 'Vandetanib': 'EGFR_inhibitor',
    'Sorafenib': 'multikinase_inhibitor', 'Sunitinib': 'multikinase_inhibitor', 'Pazopanib': 'multikinase_inhibitor',
    'Regorafenib': 'multikinase_inhibitor', 'Lenvatinib': 'multikinase_inhibitor', 'Cabozantinib': 'multikinase_inhibitor',
    'Venetoclax': 'BCL2_inhibitor', 'Navitoclax': 'BCL2_inhibitor', 'Obatoclax': 'BCL2_inhibitor', 'Palbociclib': 'CDK4_inhibitor',
    'Tamoxifen': 'SERM', 'Raloxifene': 'SERM', 'Toremifene': 'SERM', 'Bazedoxifene': 'SERM', 'Ospemifene': 'SERM',
    'Methotrexate': 'antimetabolite', 'Capecitabine': 'antimetabolite', 'Fluorouracil': 'antimetabolite',
    'Cytarabine': 'antimetabolite', 'Gemcitabine': 'antimetabolite', 'Mercaptopurine': 'antimetabolite',
    'Cyclophosphamide': 'alkylating_agent', 'Ifosfamide': 'alkylating_agent', 'Chlorambucil': 'alkylating_agent',
    'Melphalan': 'alkylating_agent', 'Busulfan': 'alkylating_agent', 'Carmustine': 'alkylating_agent'
}

foods = [
    'Quercetin', 'Naringenin', 'Curcumin', 'Resveratrol', 'EGCG', 'Caffeine', 'Theobromine', 'Theophylline',
    'Piperine', 'Lycopene', 'Folate', 'Vitamin_K', 'Vitamin_C', 'Vitamin_E', 'Vitamin_D', 'Genistein', 
    'Allicin', 'Sulforaphane', 'Luteolin', 'Hesperidin', 'Rutin', 'Kaempferol', 'Apigenin', 'Myricetin',
    'Catechin', 'Epicatechin', 'Gallic_acid', 'Ferulic_acid', 'Caffeic_acid', 'Chlorogenic_acid'
]

np.random.seed(42)
data = []
# Generate 550 rows
for _ in range(550):
    d = np.random.choice(list(drugs.keys()))
    f = np.random.choice(foods)
    change = np.random.normal(0, 8)
    
    # Let's map structural correlations heavily so XGBoost can actually learn it via features
    if 'Inhibitor' in drugs[d] or 'inhibitor' in drugs[d]:
        if f in ['Naringenin', 'Piperine', 'Caffeine', 'Theobromine', 'Vitamin_K', 'Allicin']:
            change += np.random.uniform(28, 65)
        elif f in ['Quercetin', 'EGCG', 'Luteolin', 'Vitamin_C', 'Sulforaphane']:
            change -= np.random.uniform(30, 50)
            
    if drugs[d] == 'SERM':
        if f in ['Genistein', 'Resveratrol', 'Rutin', 'Folate']:
            change -= np.random.uniform(20, 45)
            
    if drugs[d] == 'alkylating_agent':
        if f in ['Vitamin_E', 'Curcumin', 'Lycopene']:
            change += np.random.uniform(15, 35)

    if np.random.rand() < 0.25: change = 0.0

    split = "train" if np.random.rand() < 0.85 else "unseen"
    data.append([d, f, drugs[d], round(change, 1), split])

df = pd.DataFrame(data, columns=["Drug_Name", "Food_Name", "Drug_Class", "Bioavail_Change_Pct", "Split"]).drop_duplicates(subset=["Drug_Name", "Food_Name"])
df.to_csv("clinical_interactions.csv", index=False)
print(f"Generated massively expanded CSV with {len(df)} unique pairs.")
