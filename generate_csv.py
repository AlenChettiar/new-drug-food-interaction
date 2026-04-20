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

data = [[d, cls] for d, cls in drugs.items()]
df = pd.DataFrame(data, columns=["Drug_Name", "Drug_Class"])
df.to_csv("clinical_interactions.csv", index=False)
print(f"Generated base CSV with {len(df)} unique base drugs.")
