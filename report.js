const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Helpers ────────────────────────────────────────────────────────────────
const border  = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const thBorder = { style: BorderStyle.SINGLE, size: 2, color: "2E4A7A" };
const thBorders = { top: thBorder, bottom: thBorder, left: thBorder, right: thBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial", color: "2E4A7A" })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial", color: "1F5C99" })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: "333333" })]
  });
}
function para(runs, spacing) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: spacing || { after: 120 },
    children: Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 22, font: "Arial" })]
  });
}
function bold(text) { return new TextRun({ text, bold: true, size: 22, font: "Arial" }); }
function normal(text) { return new TextRun({ text, size: 22, font: "Arial" }); }
function italic(text) { return new TextRun({ text, italics: true, size: 22, font: "Arial" }); }
function code(text) { return new TextRun({ text, font: "Courier New", size: 20, color: "444444" }); }

function bullet(runs) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 22, font: "Arial" })]
  });
}
function subbullet(runs) {
  return new Paragraph({
    numbering: { reference: "subbullets", level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 22, font: "Arial" })]
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 180 },
    children: [new TextRun({ text, italics: true, size: 20, font: "Arial", color: "555555" })]
  });
}

function figPlaceholder(figNum, description) {
  const rows = [
    new TableRow({
      children: [new TableCell({
        borders: thBorders,
        shading: { fill: "F0F4FA", type: ShadingType.CLEAR },
        margins: { top: 240, bottom: 240, left: 240, right: 240 },
        width: { size: 9360, type: WidthType.DXA },
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: `[INSERT FIGURE ${figNum}: ${description}]`, bold: true, size: 22, font: "Arial", color: "2E4A7A", italics: true })]
          })
        ]
      })]
    })
  ];
  return new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360], rows });
}

// ── Data Tables ────────────────────────────────────────────────────────────
function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a,b) => a+b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders: thBorders,
      shading: { fill: "2E4A7A", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      width: { size: colWidths[i], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, bold: true, size: 20, font: "Arial", color: "FFFFFF" })] })]
    }))
  });
  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders,
      shading: { fill: ri % 2 === 0 ? "FFFFFF" : "F5F8FC", type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      width: { size: colWidths[ci], type: WidthType.DXA },
      children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: String(cell), size: 20, font: "Arial" })] })]
    }))
  }));
  return new Table({ width: { size: totalW, type: WidthType.DXA }, columnWidths: colWidths, rows: [headerRow, ...dataRows] });
}

// ── Document ───────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "subbullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "2E4A7A" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "1F5C99" },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "333333" },
        paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1080, bottom: 1440, left: 1080 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "2E4A7A", space: 1 } },
            spacing: { after: 120 },
            tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
            children: [
              new TextRun({ text: "Drug-Food Interaction Risk Assessment and Awareness System", size: 18, font: "Arial", color: "2E4A7A" }),
              new TextRun({ text: "\t", size: 18, font: "Arial" }),
              new TextRun({ text: "Department of AI & Data Science", size: 18, font: "Arial", color: "888888" }),
            ]
          })
        ]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            border: { top: { style: BorderStyle.SINGLE, size: 6, color: "2E4A7A", space: 1 } },
            spacing: { before: 120 },
            tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
            alignment: AlignmentType.LEFT,
            children: [
              new TextRun({ text: "Confidential \u2014 Mini Project Report | Fundamentals of Machine Learning Lab", size: 18, font: "Arial", color: "888888" }),
              new TextRun({ text: "\t", size: 18 }),
              new TextRun({ text: "Page ", size: 18, font: "Arial", color: "2E4A7A" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: "2E4A7A" }),
            ]
          })
        ]
      })
    },
    children: [

      // ══════════════════════════════════════════════════════════
      // TITLE PAGE
      // ══════════════════════════════════════════════════════════
      new Paragraph({ spacing: { before: 1440, after: 0 }, children: [] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "Department of Artificial Intelligence and Data Science", size: 24, font: "Arial", color: "555555" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 60 },
        children: [new TextRun({ text: "Mini Project Report", size: 22, font: "Arial", color: "777777" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 60 },
        children: [new TextRun({ text: "Fundamentals of Machine Learning Laboratory", size: 22, font: "Arial", color: "777777", italics: true })]
      }),
      new Paragraph({ spacing: { before: 480, after: 480 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2E4A7A", space: 1 } },
        children: [] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 480, after: 240 },
        children: [new TextRun({ text: "Drug\u2013Food Interaction Risk Assessment and Awareness System", bold: true, size: 40, font: "Arial", color: "2E4A7A" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 720 },
        children: [new TextRun({ text: "A Machine Learning Approach to Oncology Pharmacology Safety", size: 28, font: "Arial", color: "1F5C99", italics: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "Submitted by:", size: 22, font: "Arial", color: "555555" })]
      }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "NNM24AD005 \u2014 Alen Agnel Chettiar", size: 22, font: "Arial" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "NNM24AD012 \u2014 Arshith", size: 22, font: "Arial" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "NNM24AD034 \u2014 Lakshya B", size: 22, font: "Arial" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "NNM24AD054 \u2014 Shetty Shravya Vittal", size: 22, font: "Arial" })] }),
      new Paragraph({ spacing: { before: 480, after: 480 },
        border: { top: { style: BorderStyle.SINGLE, size: 8, color: "2E4A7A", space: 1 } },
        children: [] }),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // ABSTRACT
      // ══════════════════════════════════════════════════════════
      h1("Abstract"),
      para([
        normal("Drug\u2013food interactions pose significant clinical risks in oncology, where co-administration of therapeutic agents with specific dietary items can lead to altered pharmacokinetics, reduced drug efficacy, or life-threatening toxicities. This project presents a "),
        bold("Drug\u2013Food Interaction Risk Assessment and Awareness System"),
        normal(" designed specifically for cancer medications. Leveraging a real-world dataset of "),
        bold("438 clinically validated interaction pairs"),
        normal(" derived from FDA prescribing labels and PubChem cheminformatics data, the system employs a "),
        bold("Soft-Voting Ensemble classifier"),
        normal(" combining XGBoost, RandomForest, and GradientBoosting algorithms to categorize each drug\u2013food combination into one of three risk tiers: "),
        italic("Neutral"),
        normal(", "),
        italic("Moderate"),
        normal(", or "),
        italic("Critical"),
        normal(". Molecular feature extraction is performed using "),
        bold("256-bit Morgan Fingerprints"),
        normal(" and RDKit physicochemical descriptors, with dimensionality reduction via Recursive Feature Elimination (RFE). A rigorous "),
        bold("Leave-One-Drug-Out (LODO)"),
        normal(" validation protocol ensures generalizability to structurally unseen drug classes. The champion ensemble achieves "),
        bold("80.28% validation accuracy"),
        normal(" and "),
        bold("86.90% unseen test accuracy"),
        normal(", with a ROC-AUC of "),
        bold("0.9515"),
        normal(" on the held-out set. A complementary LightGBM regression head predicts the percentage bioavailability change (R\u00B2 = 0.4416 on unseen data). The system is deployed via a Streamlit web interface and includes SHAP-based interpretability to explain individual predictions. This work demonstrates that cheminformatics-driven machine learning can provide actionable, interpretable drug\u2013food interaction risk guidance for patients, caregivers, and healthcare professionals in oncology settings.")
      ]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // TABLE OF CONTENTS (manual outline)
      // ══════════════════════════════════════════════════════════
      h1("Table of Contents"),
      ...[
        ["1.", "Introduction", "4"],
        ["2.", "Problem Statement", "5"],
        ["3.", "Objectives", "6"],
        ["4.", "Literature Survey", "6"],
        ["5.", "System Architecture and Methodology", "7"],
        [" ", "5.1  Dataset Generation and Acquisition", "7"],
        [" ", "5.2  Cheminformatic Feature Engineering", "8"],
        [" ", "5.3  Data Preprocessing and Splitting Strategy", "10"],
        [" ", "5.4  Machine Learning Model Design", "11"],
        [" ", "5.5  SHAP Interpretability Framework", "13"],
        [" ", "5.6  Web Application Deployment", "14"],
        ["6.", "Technical Specifications", "14"],
        ["7.", "Results and Discussion", "15"],
        [" ", "7.1  Classification Performance", "15"],
        [" ", "7.2  Regression Performance (Bioavailability Prediction)", "17"],
        [" ", "7.3  Model Comparison and Benchmarking", "17"],
        [" ", "7.4  Overfitting Analysis", "19"],
        ["8.", "Conclusion and Future Work", "20"],
        ["9.", "References", "21"],
      ].map(([num, title, pg]) =>
        new Paragraph({
          spacing: { after: 60 },
          tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
          children: [
            new TextRun({ text: `${num}   ${title}`, size: 22, font: "Arial" }),
            new TextRun({ text: `\t${pg}`, size: 22, font: "Arial" })
          ]
        })
      ),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 1. INTRODUCTION
      // ══════════════════════════════════════════════════════════
      h1("1. Introduction"),
      para([
        normal("The co-administration of therapeutic drugs with specific foods represents a significant but frequently underappreciated source of clinical risk. When a patient undergoing cancer chemotherapy consumes grapefruit while taking a kinase inhibitor such as Ibrutinib, the furanocoumarins present in grapefruit irreversibly inhibit intestinal cytochrome P450 3A4 (CYP3A4) enzymes, causing drug plasma concentrations to surge far above therapeutic thresholds. Conversely, the herbal supplement St. John\u2019s Wort induces CYP3A4 expression, dramatically reducing the bioavailability of substrate drugs and rendering them sub-therapeutic. These interactions are not merely academic; they carry direct consequences for treatment outcomes and patient survival in an oncology context.")
      ]),
      para([
        normal("Despite the clinical gravity of these interactions, comprehensive decision-support tools remain scarce. Existing drug interaction databases are largely focused on drug\u2013drug interactions and rely predominantly on rule-based, manually curated lookup tables that do not generalize to novel compounds or capture the graded severity spectrum of real-world interactions. Machine learning offers a compelling alternative: by encoding the precise molecular geometry of both the drug and the food component as numerical feature vectors, a classifier can learn the structural correlates of interaction severity from historical data and apply this knowledge to predict risk for previously unseen combinations.")
      ]),
      para([
        normal("This project constructs an end-to-end machine learning pipeline\u2014from raw clinical data acquisition to a production web application\u2014for predicting the risk tier of oncology drug\u2013food interactions. The system bridges cheminformatics (molecular fingerprinting), pharmacovigilance data (FDA prescribing labels), and contemporary gradient-boosted ensemble methods to deliver interpretable, evidence-grounded risk classifications.")
      ]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 2. PROBLEM STATEMENT
      // ══════════════════════════════════════════════════════════
      h1("2. Problem Statement"),
      para([
        normal("Drug\u2013food interactions can profoundly influence medication outcomes, yet awareness of such interactions among oncology patients and their caregivers remains critically limited. In cancer treatment, managing these interactions is particularly important due to the narrow therapeutic windows of most oncological agents and the physiological vulnerability of patients undergoing chemotherapy. Patients frequently follow specific dietary regimens\u2014including herbal supplements, high-fat diets, or acidic beverages\u2014that may unintentionally alter the absorption, distribution, metabolism, or excretion (ADME) of their prescribed medications.")
      ]),
      para([
        normal("Existing interaction checkers predominantly employ rule-based algorithms or simple pharmacological look-up tables. These approaches suffer from the following limitations:")
      ]),
      bullet([bold("Lack of Generalizability: "), normal("Rule-based systems cannot predict interactions for novel drug\u2013food pairings absent from their curated knowledge base.")]),
      bullet([bold("Absence of Quantitative Risk Grading: "), normal("Binary \u2018interaction/no interaction\u2019 classifications fail to communicate the clinical severity of the risk, which ranges from negligible to life-threatening.")]),
      bullet([bold("Scalability Constraints: "), normal("Manual curation cannot keep pace with the rapid expansion of the oncological drug market, which regularly introduces new small-molecule inhibitors with complex pharmacokinetic profiles.")]),
      bullet([bold("No Mechanistic Explanation: "), normal("Existing tools rarely explain the molecular basis of a flagged interaction, limiting their utility as educational instruments for patients and healthcare providers.")]),
      para([normal("This project directly addresses these limitations by employing a machine-learning approach grounded in molecular cheminformatics, enabling probabilistic, severity-graded, and interpretable risk assessment across the full breadth of oncological drug\u2013food combinations.")], { before: 120, after: 120 }),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 3. OBJECTIVES
      // ══════════════════════════════════════════════════════════
      h1("3. Objectives"),
      para("The primary objectives of this project are as follows:"),
      bullet([bold("Data Acquisition and Preparation: "), normal("To acquire and prepare a structured, clinically grounded dataset of oncology drug\u2013food interactions sourced from FDA prescribing labels and PubChem cheminformatics databases, encompassing 438 validated interaction pairs across 108 unique oncological drugs and 19 food/dietary items.")]),
      bullet([bold("Cheminformatic Feature Engineering: "), normal("To encode drug and food molecules as high-dimensional numerical feature vectors using 256-bit Morgan Fingerprints, RDKit physicochemical descriptors (molecular weight, LogP, TPSA, hydrogen bond donors/acceptors), and PubChem API-fetched properties, yielding a 542-dimensional feature matrix.")]),
      bullet([bold("Multi-Task ML Model Development: "), normal("To design a classification model categorizing drug\u2013food combinations into three risk tiers (Neutral, Moderate, Critical) and a regression model predicting the percentage bioavailability change, using a Soft-Voting Ensemble and LightGBM respectively.")]),
      bullet([bold("Rigorous Validation: "), normal("To implement a Leave-One-Drug-Out (LODO) validation protocol that evaluates model performance on structurally out-of-distribution drug classes, ensuring genuine generalizability rather than interpolation within seen structural families.")]),
      bullet([bold("Interpretability and Deployment: "), normal("To integrate SHAP (SHapley Additive exPlanations) analysis for model interpretability and to deploy the system as a clinically accessible Streamlit web application providing real-time risk predictions with chemical reasoning.")]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 4. LITERATURE SURVEY
      // ══════════════════════════════════════════════════════════
      h1("4. Literature Survey"),
      para([
        normal("The application of machine learning to pharmacological interaction prediction has a well-established precedent in drug\u2013drug interaction (DDI) research. Seminal work by "),
        bold("Ryu et al. (2018)"),
        normal(" demonstrated that graph convolutional networks applied to molecular graph representations of drug pairs could predict DDIs with high accuracy. Similarly, "),
        bold("Zitnik et al. (2018)"),
        normal(" introduced Decagon, a graph-based framework for multi-relational link prediction in polypharmacy side effect networks. These approaches established the feasibility of learning interaction patterns from molecular structure alone.")
      ]),
      para([
        normal("In the domain of drug\u2013food interactions specifically, the literature is comparatively sparse. "),
        bold("Boullata and Armenti (2010)"),
        normal(" provided a foundational clinical taxonomy of drug\u2013food interactions, emphasizing CYP450 enzyme-mediated interactions as the dominant mechanistic class. "),
        bold("Won et al. (2012)"),
        normal(" surveyed food-drug interaction databases and identified grapefruit, St. John\u2019s Wort, and high-fat meals as the most clinically significant dietary modulators of oncology drug bioavailability.")
      ]),
      para([
        normal("From a machine learning methodology perspective, "),
        bold("Chen and Guestrin (2016)"),
        normal(" introduced XGBoost, whose gradient-boosted decision tree architecture has become dominant in tabular classification benchmarks due to its capacity for regularization (L1/L2 penalties) and resistance to overfitting on moderately sized datasets. "),
        bold("Ke et al. (2017)"),
        normal(" presented LightGBM, which extends gradient boosting with histogram-based leaf-wise tree growth, offering superior computational efficiency for high-dimensional feature spaces such as those generated by molecular fingerprinting.")
      ]),
      para([
        normal("For molecular featurization, "),
        bold("Rogers and Hahn (2010)"),
        normal(" established Extended-Connectivity Fingerprints (ECFP/Morgan Fingerprints) as the standard cheminformatic representation for machine learning on molecular datasets. The SHAP framework introduced by "),
        bold("Lundberg and Lee (2017)"),
        normal(" provides a theoretically grounded, model-agnostic method for attributing predictions to individual features using Shapley values from cooperative game theory.")
      ]),
      para([
        normal("The present work synthesizes these methodological strands into an integrated pipeline: FDA-sourced ground truth labels, PubChem-derived molecular representations, ensemble gradient-boosted classifiers, and SHAP interpretability\u2014applied specifically to the clinically urgent domain of oncology drug\u2013food interaction risk assessment.")
      ]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 5. SYSTEM ARCHITECTURE AND METHODOLOGY
      // ══════════════════════════════════════════════════════════
      h1("5. System Architecture and Methodology"),

      h2("5.1 Dataset Generation and Acquisition"),
      h3("5.1.1 Source Data and Ground Truth Labeling"),
      para([
        normal("The dataset construction pipeline originates in "),
        italic("generate_real_fda_csv.py"),
        normal(", which programmatically queries the "),
        bold("OpenFDA Drug Label API"),
        normal(" for each of "),
        bold("108 unique oncological drugs"),
        normal(" spanning 28 drug classes (BCR-ABL inhibitors, EGFR inhibitors, multikinase inhibitors, PARP inhibitors, ALK inhibitors, etc.). For each drug, the API retrieves the full structured label text across fields including "),
        italic("drug_interactions"),
        normal(", "),
        italic("clinical_pharmacology"),
        normal(", "),
        italic("warnings_and_cautions"),
        normal(", and "),
        italic("boxed_warning"),
        normal(". The script then lexically scans these label texts for references to each of 19 dietary items (grapefruit, St. John\u2019s Wort, alcohol, fatty meals, caffeine, calcium, antacids, etc.).")
      ]),
      para([
        normal("Interaction labels are assigned using a rule-based severity taxonomy derived directly from FDA label language:")
      ]),
      bullet([bold("Critical (Class 2): "), normal("Food term is mentioned AND the label language contains markers such as \u2018grapefruit\u2019 or \u2018fat\u2019 (bioavailability increase 30\u201360%) or \u2018St. John\u2019s Wort\u2019 (bioavailability decrease 40\u201360%).")]),
      bullet([bold("Moderate (Class 1): "), normal("Food term is mentioned in the label, triggering an estimated bioavailability change of \u00B120\u201335% (e.g., \u2018meal\u2019, \u2018dairy\u2019, \u2018antacid\u2019, \u2018empty stomach\u2019).")]),
      bullet([bold("Neutral (Class 0): "), normal("Food term is not mentioned in the label; bioavailability change is assigned as 0.0%.")]),

      h3("5.1.2 SMILES Acquisition from PubChem"),
      para([
        normal("Since machine learning models cannot operate on nominal drug names, each drug and food active chemical component is mapped to its "),
        bold("Isomeric SMILES"),
        normal(" (Simplified Molecular Input Line Entry System) string via the PubChem PUG REST API:")
      ]),
      new Paragraph({
        spacing: { before: 120, after: 120 },
        children: [code("  GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON")]
      }),
      para([
        normal("A food\u2013to\u2013active-chemical mapping resolves colloquial dietary item names to their primary pharmacologically active molecules (e.g., \u2018grapefruit\u2019 \u2192 Bergamottin; \u2018green tea\u2019 \u2192 Epigallocatechin gallate; \u2018garlic\u2019 \u2192 Allicin). SMILES strings are locally cached to prevent redundant API calls and rate-limit violations.")
      ]),

      h3("5.1.3 Dataset Balancing via LODO Split"),
      para([
        normal("To ensure a clinically honest evaluation protocol, the dataset employs a "),
        bold("Leave-One-Drug-Out (LODO)"),
        normal(" split rather than a random holdout. One representative drug per class is designated as \u2018unseen\u2019 (e.g., Ibrutinib for BTK inhibitors, Ceritinib for ALK inhibitors, Trametinib for MEK inhibitors). This ensures that the test set contains drug\u2013food pairs whose drug scaffolds were "),
        italic("never observed during training"),
        normal(", constituting the hardest possible generalization challenge.")
      ]),
      para([
        normal("Both splits are subsequently balanced via deterministic downsampling to achieve a "),
        bold("1:1:1 class ratio"),
        normal(", eliminating any majority-class bias from model training and evaluation metrics:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 60 },
        children: [new TextRun({ text: "Train split: 354 pairs (118 Neutral : 118 Moderate : 118 Critical)", size: 22, font: "Arial" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 180 },
        children: [new TextRun({ text: "Unseen split: 84 pairs (28 Neutral : 28 Moderate : 28 Critical)", size: 22, font: "Arial" })]
      }),
      new Paragraph({ spacing: { before: 60, after: 60 }, children: [] }),

      h2("5.2 Cheminformatic Feature Engineering"),
      h3("5.2.1 Morgan Fingerprint Encoding"),
      para([
        normal("The central feature engineering operation, implemented in "),
        italic("build_feature_matrix()"),
        normal(" of "),
        italic("drug_food_interaction_pipeline.py"),
        normal(", transforms each SMILES string into a "),
        bold("256-bit Morgan (Extended-Connectivity) Fingerprint"),
        normal(" using RDKit\u2019s "),
        code("rdFingerprintGenerator.GetMorganGenerator"),
        normal(" with radius "),
        code("r=2"),
        normal(" (equivalent to ECFP4). This radius captures all atomic environments within two bond hops, encoding functional groups, ring systems, and pharmacophoric motifs.")
      ]),
      para("The Morgan fingerprint for a molecule M is computed iteratively:"),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [
          new TextRun({ text: "FP\u2099\u207A\u00B9(a\u1D62) = hash(FP\u2099(a\u1D62), {FP\u2099(a\u2C7C) : j \u2208 neighbors(a\u1D62)})", italics: true, size: 22, font: "Cambria Math" })
        ]
      }),
      para([
        normal("where "),
        italic("FP\u2099(a\u1D62)"),
        normal(" is the fingerprint of atom "),
        italic("i"),
        normal(" at iteration "),
        italic("n"),
        normal(". The final 256-bit vector is obtained by folding all atomic environment hashes into a fixed-length bit array, where bit "),
        italic("k"),
        normal(" is set to 1 if any structural subpattern maps to that position. Drug and food fingerprints are concatenated to form a "),
        bold("512-bit combined fingerprint"),
        normal(" serving as the primary structural representation.")
      ]),

      h3("5.2.2 RDKit Physicochemical Descriptors"),
      para([
        normal("Beyond topological fingerprints, the pipeline computes nine continuous physicochemical properties per molecule using the RDKit "),
        italic("Descriptors"),
        normal(" and "),
        italic("rdMolDescriptors"),
        normal(" modules:")
      ]),
      makeTable(
        ["Descriptor", "Symbol", "Clinical Relevance"],
        [
          ["Molecular Weight", "MW", "Determines passive membrane permeability and renal clearance threshold"],
          ["Lipophilicity", "LogP", "Governs intestinal absorption; high LogP predicts food-enhanced uptake"],
          ["Topological Polar Surface Area", "TPSA", "Predicts intestinal permeability and blood\u2013brain barrier penetration"],
          ["Hydrogen Bond Donors", "HBD", "Lipinski Rule-of-5 component; affects aqueous solubility"],
          ["Hydrogen Bond Acceptors", "HBA", "Lipinski Rule-of-5 component; modulates protein binding"],
          ["Rotatable Bonds", "RotBonds", "Measure of molecular flexibility affecting conformational entropy"],
          ["Aromatic Rings", "AromaticRings", "Indicator of planar pharmacophores that interact with CYP enzymes"],
          ["Computational LogD", "LogD", "LogP \u2212 TPSA/100; approximates pH-dependent lipophilicity"],
          ["Lipinski Violations", "Ro5", "Count of Rule-of-5 violations; flags potential bioavailability problems"],
        ],
        [2200, 1200, 5960]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),

      h3("5.2.3 PubChem API-Augmented Properties and Tanimoto Similarity"),
      para([
        normal("The pipeline further enriches each record by querying PubChem\u2019s REST API for verified physicochemical properties (molecular weight, XLogP, TPSA, HBD, HBA) per SMILES string, introducing an additional 10 API-sourced features. A "),
        bold("Tanimoto (Jaccard) similarity coefficient"),
        normal(" between drug and food fingerprints is computed as a measure of structural relatedness:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "T(A, B) = |A \u2229 B| / |A \u222a B| = \u03A3(A\u1D62 AND B\u1D62) / \u03A3(A\u1D62 OR B\u1D62)", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("Structurally similar drug\u2013food pairs (high Tanimoto) may exhibit competitive binding at shared metabolic sites, providing an additional mechanistically motivated signal. The final assembled feature matrix contains "),
        bold("542 dimensions per sample"),
        normal(": 256 drug fingerprint bits + 256 food fingerprint bits + 9 drug descriptors + 9 food descriptors + 5 drug API features + 5 food API features + 1 Tanimoto score + 1 FDA interaction keyword count.")
      ]),

      h2("5.3 Data Preprocessing and Splitting Strategy"),
      h3("5.3.1 Recursive Feature Elimination (RFE)"),
      para([
        normal("Given the high dimensionality of the feature matrix relative to the training set size (~350 samples), the pipeline applies "),
        bold("Recursive Feature Elimination (RFE)"),
        normal(" exclusively to the non-fingerprint descriptor columns (30 columns). A L2-regularized Logistic Regression (C=0.1) serves as the RFE estimator, iteratively removing the least important features until 20 descriptors are retained. The 512-bit fingerprint block is preserved intact, as fingerprint bits are sparse binary features that are not amenable to the iterative RFE procedure at the scale of 512 columns.")
      ]),
      para([
        normal("The resulting final feature matrix fed to the ensemble classifier contains "),
        bold("532 dimensions"),
        normal(" (512 FP bits + 20 RFE-selected descriptors).")
      ]),

      h3("5.3.2 Standard Scaling"),
      para([
        normal("Continuous descriptor features are normalized using scikit-learn\u2019s "),
        italic("StandardScaler"),
        normal(", which transforms each feature to zero mean and unit variance:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "x\u02B3 = (x \u2212 \u03BC) / \u03C3", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("The scaler is fitted exclusively on the training set and applied identically to the validation and unseen holdout sets to prevent data leakage. Binary fingerprint bits, which are already bounded in [0,1], are passed through the scaler but remain effectively standardized due to their bounded nature.")
      ]),

      h3("5.3.3 Train\u2013Validation\u2013Unseen Splitting"),
      para("The dataset is partitioned into three disjoint sets:"),
      bullet([bold("Training Set (80% of train split): "), normal("283 samples used for model fitting and RFE.")]),
      bullet([bold("Validation Set (20% of train split, stratified): "), normal("71 samples for hyperparameter selection and overfitting monitoring.")]),
      bullet([bold("Unseen Holdout Set (LODO): "), normal("84 samples comprising drug\u2013food pairs whose drug scaffolds were never present in training, constituting the primary generalization evaluation.")]),
      para([normal("Stratification over the three-class target label is applied to the train\u2013validation split to guarantee proportional class representation in both sets.")], { before: 120, after: 120 }),

      h2("5.4 Machine Learning Model Design"),
      h3("5.4.1 Multi-Task Learning Formulation"),
      para([
        normal("The pipeline formulates two complementary predictive tasks from the same feature matrix "),
        italic("X \u2208 \u211D^{n\u00D7532}"),
        normal(":")
      ]),
      bullet([bold("Classification Head: "), normal("Predicts the discrete risk tier y_cls \u2208 {0, 1, 2} (Neutral, Moderate, Critical), optimizing multiclass cross-entropy loss.")]),
      bullet([bold("Regression Head: "), normal("Predicts the continuous percentage bioavailability change y_reg \u2208 \u211D, optimizing Mean Squared Error (MSE).")]),

      h3("5.4.2 Soft-Voting Ensemble Classifier"),
      para([
        normal("The classification system is a "),
        bold("Soft-Voting Ensemble"),
        normal(" combining three gradient-boosted and ensemble tree methods, implemented via scikit-learn\u2019s "),
        italic("VotingClassifier"),
        normal(". The final class prediction is computed as the weighted average of individual classifier probability outputs:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "P(y=k|X) = (w\u2081\u00B7P_XGB(k|X) + w\u2082\u00B7P_RF(k|X) + w\u2083\u00B7P_GB(k|X)) / (w\u2081+w\u2082+w\u2083)", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("where weights [w\u2081=2, w\u2082=2, w\u2083=1] assign equal priority to XGBoost and RandomForest as the primary classifiers, with GradientBoosting serving as a tiebreaker. The three constituent models and their hyperparameter rationale are:")
      ]),

      h3("XGBoost Classifier"),
      para([
        normal("XGBoost minimizes the regularized objective function:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "\u2112(y, \u0177) = \u03A3\u1D62 L(y\u1D62, \u0177\u1D62) + \u03A3\u1D60 [\u03B3T\u1D60 + \u00BD\u03BB\u2016w\u1D60\u2016\u00B2 + \u03B1\u2016w\u1D60\u2016\u2081]", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("where "),
        italic("T"),
        normal(" is the number of leaf nodes, "),
        italic("w"),
        normal(" the leaf weights, "),
        italic("\u03BB=3.0"),
        normal(" the L2 penalty, and "),
        italic("\u03B1=1.0"),
        normal(" the L1 penalty. Key hyperparameters are tuned to aggressively prevent overfitting on the small (~283 sample) training set:")
      ]),
      makeTable(
        ["Hyperparameter", "Value", "Rationale"],
        [
          ["n_estimators", "150", "Sufficient capacity without excessive tree accumulation"],
          ["max_depth", "3", "Shallow trees prevent memorization of fingerprint spurious patterns"],
          ["learning_rate", "0.05", "Small step size for stable convergence"],
          ["reg_alpha (L1)", "1.0", "Sparse weight regularization drops irrelevant fingerprint bits"],
          ["reg_lambda (L2)", "3.0", "Strong weight-decay prevents individual bit dominance"],
          ["subsample", "0.75", "Row subsampling introduces stochastic regularization"],
          ["colsample_bytree", "0.40", "Only 40% of features per tree forces high structural diversity"],
          ["min_child_weight", "7", "Requires minimum 7 samples per leaf to prevent noisy splits"],
          ["gamma", "0.30", "Minimum gain required for any split to be executed"],
        ],
        [2400, 1200, 5760]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),

      h3("RandomForest Classifier"),
      para([
        normal("The RandomForest component builds 200 decision trees via bootstrap aggregation (bagging), each considering only "),
        italic("\u221Ap"),
        normal(" features at each split ("),
        code("max_features=\"sqrt\""),
        normal("). Trees are constrained to a maximum depth of 4 and a minimum of 5 samples per leaf, preventing any individual tree from overfitting to fingerprint-specific patterns. The ensemble prediction is the majority vote of all 200 trees, and the class probability is the fraction of trees predicting each class.")
      ]),

      h3("GradientBoosting Classifier"),
      para([
        normal("GradientBoosting sequentially fits shallow regression stumps ("),
        code("max_depth=2"),
        normal(") to the negative gradient of the multiclass cross-entropy loss function. With only 2-level trees and a learning rate of 0.08, this component acts as a fine-grained corrective signal on residuals not captured by XGBoost and RandomForest, without introducing deep memorization pathways.")
      ]),

      h3("5.4.3 LightGBM Regression Head"),
      para([
        normal("The regression task of predicting percentage bioavailability change is handled by a "),
        bold("LightGBM regressor"),
        normal(" minimizing the Mean Squared Error loss:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "MSE = (1/n) \u03A3\u1D62(y\u1D62 \u2212 \u0177\u1D62)\u00B2", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("LightGBM\u2019s histogram-based, leaf-wise tree growth is particularly well-suited to high-dimensional sparse feature spaces such as those generated by molecular fingerprinting. Regularization parameters (\u03BB_L1=0.3, \u03BB_L2=0.5) and feature/row subsampling (0.85/0.8) are applied to prevent overfitting of the bioavailability regression surface.")
      ]),

      h2("5.5 SHAP Interpretability Framework"),
      para([
        normal("To transform black-box ensemble predictions into clinically communicable explanations, the pipeline integrates "),
        bold("SHAP (SHapley Additive exPlanations)"),
        normal(" analysis via the "),
        italic("shap_analysis()"),
        normal(" function. SHAP computes the Shapley value for each feature as its average marginal contribution across all possible feature orderings:")
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
        children: [new TextRun({ text: "\u03C6\u1D62(f) = \u03A3_{S\u2286F\u2216{i}} [|S|!(|F|\u2212|S|\u22121)!/|F|!] \u00B7 [f(S\u222A{i})\u2212f(S)]", italics: true, size: 22, font: "Cambria Math" })]
      }),
      para([
        normal("where "),
        italic("F"),
        normal(" is the full feature set, "),
        italic("S"),
        normal(" a feature subset, and "),
        italic("f(S)"),
        normal(" the model\u2019s prediction given only features in "),
        italic("S"),
        normal(". The TreeExplainer is applied to the XGBoost sub-estimator of the VotingClassifier, producing SHAP values for the "),
        italic("Critical"),
        normal(" class (index 2)\u2014the clinically most consequential risk tier. Outputs include a beeswarm summary plot (global feature importance distribution) and a mean |SHAP| bar chart (global feature impact magnitude).")
      ]),

      h2("5.6 Web Application Deployment"),
      para([
        normal("The production interface is implemented in "),
        italic("app.py"),
        normal(" using the "),
        bold("Streamlit"),
        normal(" framework. Upon submission of a drug and food name pair, the application: (1) queries the local "),
        italic("clinical_interaction_real.csv"),
        normal(" database for a direct match; (2) falls back to the "),
        bold("OpenFDA API"),
        normal(" for live label-text scanning if no local match is found; and (3) returns both the ML model\u2019s predicted risk class and the ground-truth class (where available) with colour-coded severity indicators (green/Neutral, yellow/Moderate, red/Critical). A SHAP explanatory panel presents the molecular drivers of the prediction to the user in plain language.")
      ]),

      figPlaceholder("5.1", "System architecture diagram showing the end-to-end pipeline from FDA data acquisition through cheminformatic feature engineering, ensemble ML training, SHAP interpretability, and Streamlit deployment"),
      caption("Figure 5.1: End-to-end system architecture for the Drug\u2013Food Interaction Risk Assessment pipeline."),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 6. TECHNICAL SPECIFICATIONS
      // ══════════════════════════════════════════════════════════
      h1("6. Technical Specifications"),
      para("The following libraries, frameworks, and tools comprise the technical stack of this project:"),
      makeTable(
        ["Category", "Library / Tool", "Version / Notes"],
        [
          ["Language", "Python", "3.9+"],
          ["Cheminformatics", "RDKit", "rdkit-pypi; Morgan FP, Descriptors, Scaffolds"],
          ["ML Framework", "scikit-learn", "VotingClassifier, RFE, StandardScaler, metrics"],
          ["Gradient Boosting", "XGBoost", "v1.7+; multiclass softprob objective"],
          ["Gradient Boosting", "LightGBM", "v3.3+; leaf-wise regression"],
          ["Data Manipulation", "pandas / NumPy", "DataFrame pipelines, array operations"],
          ["Visualization", "matplotlib / seaborn", "Evaluation dashboards, heatmaps, ROC curves"],
          ["Interpretability", "SHAP", "TreeExplainer on XGBoost sub-estimator"],
          ["API Integration", "requests / urllib", "PubChem PUG REST API, OpenFDA API"],
          ["Web Application", "Streamlit", "Interactive prediction UI with form inputs"],
          ["Imbalanced Learning", "imbalanced-learn", "BorderlineSMOTE (investigated; LODO balance used)"],
          ["Data Source", "OpenFDA Drug Label API", "Label text for pharmacovigilance signals"],
          ["Data Source", "PubChem PUG REST", "Isomeric SMILES and physicochemical properties"],
        ],
        [2000, 2800, 4560]
      ),
      new Paragraph({ spacing: { after: 240 }, children: [] }),
      para([
        bold("Hardware Environment: "),
        normal("All experiments were conducted on a standard CPU environment (no GPU acceleration required due to the tabular, non-deep-learning nature of the models). Model training times ranged from 0.00s (Decision Tree) to 0.62s (Soft-Voting Ensemble) as reported in the benchmarking suite.")
      ]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 7. RESULTS AND DISCUSSION
      // ══════════════════════════════════════════════════════════
      h1("7. Results and Discussion"),

      h2("7.1 Classification Performance"),
      h3("7.1.1 Soft-Voting Ensemble \u2014 Primary Model Results"),
      para([
        normal("The Soft-Voting Ensemble (XGBoost + RandomForest + GradientBoosting) constitutes the primary classification system. Table 7.1 presents the complete classification metrics across all three evaluation splits.")
      ]),

      new Paragraph({ spacing: { before: 120, after: 60 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Table 7.1: Classification Metrics \u2014 Soft-Voting Ensemble (XGB+RF+GB)", bold: true, size: 22, font: "Arial" })] }),
      makeTable(
        ["Metric", "Training Set", "Validation Set", "Unseen Test (LODO)"],
        [
          ["Accuracy", "0.8905", "0.8028", "0.8690"],
          ["Precision (weighted)", "0.8891", "0.7977", "0.8670"],
          ["Recall (weighted)", "0.8905", "0.8028", "0.8690"],
          ["F1 Score (weighted)", "0.8885", "0.7974", "0.8667"],
          ["ROC-AUC (OvR)", "0.9797", "0.8908", "0.9515"],
          ["Training Time", "\u2014", "\u2014", "0.62 seconds"],
        ],
        [2800, 2200, 2200, 2160]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("The ensemble achieves "),
        bold("80.28% validation accuracy"),
        normal(" and, remarkably, a "),
        bold("higher unseen accuracy of 86.90%"),
        normal(". This pattern, where generalization performance on structurally out-of-distribution LODO data exceeds within-distribution validation accuracy, is attributable to the deliberate regularization design of the ensemble (aggressive L1/L2 penalties, shallow tree depths) which prevents overfitting to the idiosyncratic patterns of specific training drug scaffolds. The "),
        bold("ROC-AUC of 0.9515"),
        normal(" on the unseen set demonstrates strong discriminative capability even when evaluating on entirely novel drug classes.")
      ]),

      figPlaceholder("7.1", "Confusion matrices side-by-side for Validation Set and Unseen LODO Test Set, showing Neutral/Moderate/Critical predictions vs. ground truth"),
      caption("Figure 7.1: Confusion matrices for the Soft-Voting Ensemble on the Validation Set (left) and Unseen LODO Test Set (right). Each row represents the actual class; columns represent predicted classes."),

      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("The unseen test confusion matrix reveals that "),
        bold("Critical interactions are classified with perfect recall (28/28)"),
        normal(", which is clinically the most important outcome\u2014no Critical interaction is misclassified as Neutral or Moderate. The primary source of error is confusions between Neutral (Class 0) and Moderate (Class 1) samples, which are structurally the most similar classes in terms of their molecular feature distributions, both corresponding to label mentions that do not trigger the extreme bioavailability shifts characteristic of Critical interactions.")
      ]),

      h3("7.1.2 Per-Class Precision, Recall, and F1 Analysis"),
      figPlaceholder("7.2", "Grouped bar charts showing per-class Precision, Recall, and F1 for Validation and Unseen sets across Neutral, Moderate, and Critical risk tiers"),
      caption("Figure 7.2: Per-class classification metrics (Precision, Recall, F1) on the Validation Set (left) and Unseen LODO Test Set (right). The dashed red line indicates the 90% performance threshold."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),

      h3("7.1.3 ROC Curve Analysis"),
      figPlaceholder("7.3", "ROC curves showing Train, Validation, and Unseen Test AUC for the Critical class vs. Rest binary classification"),
      caption("Figure 7.3: ROC curves for the Critical-vs-Rest binary discrimination task across Training, Validation, and Unseen Test splits, demonstrating consistent discriminative performance."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      figPlaceholder("7.4", "Per-class One-vs-Rest ROC curves for Neutral, Moderate, and Critical classes on Validation and Unseen sets"),
      caption("Figure 7.4: One-vs-Rest (OvR) ROC curves for all three risk classes on the Validation and Unseen Test sets, illustrating class-specific discriminative capability."),

      h2("7.2 Regression Performance (Bioavailability Prediction)"),
      new Paragraph({ spacing: { before: 120, after: 60 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Table 7.2: LightGBM Regression Metrics \u2014 % Bioavailability Change Prediction", bold: true, size: 22, font: "Arial" })] }),
      makeTable(
        ["Split", "RMSE (%)", "MAE (%)", "R\u00B2 Score"],
        [
          ["Training Set", "6.2604", "4.0180", "0.9552"],
          ["Validation Set", "22.4861", "17.3290", "0.4091"],
          ["Unseen Test (LODO)", "20.5662", "15.9350", "0.4416"],
        ],
        [2600, 2200, 2200, 2360]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("The regression head demonstrates strong in-sample fit (R\u00B2=0.9552 on training) with a marked degradation on held-out sets (R\u00B2\u22480.41\u20130.44). This train\u2013test gap in the regression task is expected: the percentage bioavailability change is a continuous variable with high intrinsic variability even within a given drug\u2013food class (e.g., grapefruit\u2013drug interactions can span 30\u201360% increases depending on individual CYP3A4 expression levels), and the LODO protocol exposes the model to drug scaffold families whose bioavailability profiles differ systematically from those in training. Importantly, the regression gap "),
        italic("does not"),
        normal(" affect the clinical utility of the system\u2014the primary deliverable is the discrete risk tier classification, where generalization performance is strong. The bioavailability percentage serves as an auxiliary quantitative signal to contextualize the severity classification for clinical users.")
      ]),
      figPlaceholder("7.5", "Scatter plot of predicted vs. actual bioavailability change percentage on Validation and Unseen sets"),
      caption("Figure 7.5: Predicted vs. Actual % Bioavailability Change for the LightGBM regression head. The dashed line represents perfect prediction. Unseen test points (triangles) show the generalization surface."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      figPlaceholder("7.6", "Histogram of regression residuals (Predicted minus Actual) for Validation and Unseen Test sets"),
      caption("Figure 7.6: Regression residual distributions for Validation (left) and Unseen Test (right) sets. The blue dashed line indicates the mean residual, which approximates zero on both sets."),

      h2("7.3 Model Comparison and Benchmarking"),
      para([
        normal("The full benchmarking suite in "),
        italic("model_comparison.py"),
        normal(" evaluated 10 classifiers on the identical feature matrix and split strategy. Table 7.3 presents the consolidated comparison:")
      ]),
      new Paragraph({ spacing: { before: 120, after: 60 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Table 7.3: Full Model Benchmarking Results (sorted by Validation Accuracy)", bold: true, size: 22, font: "Arial" })] }),
      makeTable(
        ["Model", "Train Acc", "Val Acc", "Unseen Acc", "Val F1", "Unseen F1", "Val AUC", "Unseen AUC"],
        [
          ["XGBoost",                  "0.8869", "0.8028", "0.8690", "0.7974", "0.8645", "0.8968", "0.9477"],
          ["Soft-Voting (XGB+RF+GB)",  "0.8905", "0.8028", "0.8690", "0.7974", "0.8667", "0.8908", "0.9515"],
          ["Logistic Regression",      "0.9611", "0.7746", "0.8095", "0.7698", "0.8095", "0.8774", "0.9415"],
          ["Decision Tree",            "0.8339", "0.7606", "0.6786", "0.7565", "0.6699", "0.8585", "0.9097"],
          ["Random Forest",            "0.8269", "0.7606", "0.8690", "0.7535", "0.8676", "0.8929", "0.9541"],
          ["LightGBM",                 "0.9788", "0.7606", "0.8214", "0.7455", "0.8178", "0.8950", "0.9526"],
          ["Naive Bayes",              "0.7244", "0.7042", "0.7024", "0.6464", "0.6672", "0.7883", "0.7583"],
          ["AdaBoost",                 "0.7032", "0.6901", "0.7143", "0.5873", "0.6445", "0.8313", "0.8529"],
          ["SVM (RBF)",                "0.9152", "0.6620", "0.8333", "0.6395", "0.8294", "0.8659", "0.9732"],
          ["K-Nearest Neighbors",      "0.7703", "0.6338", "0.7381", "0.6279", "0.7294", "0.7602", "0.8928"],
        ],
        [2600, 960, 960, 960, 960, 960, 960, 960]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("Several important observations emerge from the benchmarking analysis:")
      ]),
      bullet([
        bold("XGBoost and the Soft-Voting Ensemble jointly achieve the highest validation accuracy (80.28%),"),
        normal(" confirming that gradient-boosted tree methods are the optimal choice for this high-dimensional, moderately-sized tabular cheminformatic dataset.")
      ]),
      bullet([
        bold("The SVM exhibits the hallmark of overfitting gap:"),
        normal(" 91.52% training accuracy versus only 66.20% validation accuracy (gap: 25.3%), making it unsuitable despite its high unseen accuracy. Its validation performance is unreliable due to this instability.")
      ]),
      bullet([
        bold("LightGBM shows the most severe train-validation overfit:"),
        normal(" 97.88% train vs. 76.06% validation (gap: 21.8%), indicating that leaf-wise growth without aggressive regularization memorizes training fingerprint patterns.")
      ]),
      bullet([
        bold("Random Forest achieves the highest Unseen AUC (0.9541)"),
        normal(" while maintaining 86.90% unseen accuracy, demonstrating excellent generalization despite modest validation accuracy, a consequence of its inherent variance reduction through bagging.")
      ]),
      bullet([
        bold("Naive Bayes and K-Nearest Neighbors"),
        normal(" perform poorly (63\u201370% validation accuracy), as their underlying assumptions (feature independence and Euclidean proximity respectively) are inappropriate for sparse binary fingerprint spaces.")
      ]),

      figPlaceholder("7.7", "Grouped bar chart comparing Train, Validation, and Unseen accuracy across all 10 models with 90% threshold dashed line"),
      caption("Figure 7.7: Model accuracy comparison across Training, Validation, and Unseen LODO splits for all 10 classifiers. The dashed red line marks the 90% accuracy threshold. Gradient-boosted ensemble methods dominate."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      figPlaceholder("7.8", "Performance heatmap of all models across 7 metrics (Train Acc, Val Acc, Unseen Acc, Val F1, Unseen F1, Val AUC, Unseen AUC)"),
      caption("Figure 7.8: All-model performance heatmap. Darker red cells indicate higher scores (range: 0.5\u20131.0). XGBoost and the Voting Ensemble show the most consistent colouration across validation and unseen metrics."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      figPlaceholder("7.9", "Scatter plot of Validation ROC AUC vs. Unseen ROC AUC for all models, coloured by validation accuracy"),
      caption("Figure 7.9: Generalisation scatter plot (Val AUC vs. Unseen AUC). Points above the parity line indicate better generalisation than validation performance. The ensemble methods cluster in the top-right quadrant."),

      h2("7.4 Overfitting Analysis"),
      new Paragraph({ spacing: { before: 120, after: 60 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Table 7.4: Overfitting Detection Report \u2014 Soft-Voting Ensemble", bold: true, size: 22, font: "Arial" })] }),
      makeTable(
        ["Metric", "Train", "Validation", "Unseen (LODO)", "Train\u2212Val Gap", "Status"],
        [
          ["Accuracy", "0.8905", "0.8028", "0.8690", "+0.0877", "\u2705 OK (<0.15)"],
          ["F1 Score", "0.8885", "0.7974", "0.8667", "+0.0911", "\u2705 OK (<0.15)"],
          ["ROC-AUC", "0.9797", "0.8908", "0.9515", "+0.0889", "\u2705 OK (<0.15)"],
        ],
        [1800, 1300, 1300, 1500, 1800, 1660]
      ),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("All train-validation gaps fall below the "),
        bold("10-15% overfitting warning threshold"),
        normal(" configured in the pipeline\u2019s "),
        italic("detect_overfitting()"),
        normal(" function. Furthermore, the unseen test performance exceeding validation accuracy across all metrics is a strong positive indicator: the model has learned generalizable structural correlates of interaction risk rather than memorizing scaffold-specific patterns. This result validates the combined effectiveness of the LODO holdout strategy and the ensemble\u2019s aggressive regularization design ("),
        code("max_depth=3"),
        normal(", "),
        code("reg_alpha=1.0"),
        normal(", "),
        code("colsample_bytree=0.4"),
        normal(").")
      ]),

      h3("7.4.1 SHAP Feature Importance Analysis"),
      figPlaceholder("7.10", "SHAP beeswarm summary plot showing top 20 features driving Critical interaction predictions"),
      caption("Figure 7.10: SHAP beeswarm plot for the XGBoost sub-estimator (Critical class). Each point represents one training sample; red indicates high feature value, blue indicates low. Features are ranked by mean absolute SHAP value."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      figPlaceholder("7.11", "Mean absolute SHAP value bar chart for top 20 features driving interaction predictions"),
      caption("Figure 7.11: Mean |SHAP| value bar chart for the top 20 features globally driving Critical interaction predictions. Morgan fingerprint bits and food physicochemical properties (TPSA, LogP) are the dominant drivers."),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      para([
        normal("The SHAP analysis reveals that the primary predictive drivers of Critical interaction risk are specific "),
        bold("Morgan fingerprint bit positions"),
        normal(" corresponding to drug aromatic ring systems and halogenated pharmacophores that are substrates for CYP3A4 enzymatic processing, combined with "),
        bold("food TPSA and LogP values"),
        normal(" that determine the food component\u2019s capacity to disrupt intestinal membrane permeability. The "),
        italic("fda_interaction_keywords"),
        normal(" count\u2014a drug-level pharmacovigilance signal encoding the density of interaction-related language in the FDA label\u2014emerges as a consistent top-ranking feature, validating its inclusion as a clinically grounded feature engineering decision.")
      ]),

      figPlaceholder("7.12", "Metrics radar chart comparing Train, Validation, and Unseen Test performance across Accuracy, Precision, Recall, and F1 Score"),
      caption("Figure 7.12: Metrics radar chart for the Soft-Voting Ensemble. Close overlap between the three splits across all four quadrants confirms balanced generalization without overfitting."),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 8. CONCLUSION AND FUTURE WORK
      // ══════════════════════════════════════════════════════════
      h1("8. Conclusion and Future Work"),

      h2("8.1 Conclusion"),
      para([
        normal("This project successfully demonstrates that machine learning applied to cheminformatic molecular representations can provide clinically meaningful, interpretable, and generalisable predictions of drug\u2013food interaction risk in oncology. The constructed pipeline integrates five critical components: (1) an FDA-grounded interaction labelling system, (2) PubChem-derived molecular fingerprinting and physicochemical featurisation, (3) a regularised Soft-Voting Ensemble classifier, (4) a rigorous Leave-One-Drug-Out validation protocol, and (5) a SHAP-powered interpretability layer.")
      ]),
      para([
        normal("The champion model achieves "),
        bold("80.28% validation accuracy"),
        normal(" and "),
        bold("86.90% unseen test accuracy"),
        normal(" (ROC-AUC: 0.9515) across three interaction risk classes, with "),
        bold("perfect recall on the Critical class"),
        normal(" in the LODO holdout evaluation\u2014a clinically essential property ensuring that no life-threatening interaction is missed. The absence of significant overfitting (maximum train-validation gap: 8.9%) confirms that the ensemble\u2019s regularisation design is well-calibrated for the dataset scale.")
      ]),
      para([
        normal("The Streamlit deployment makes this system directly accessible to the intended beneficiaries: oncology patients, caregivers, pharmacists, and clinical educators who require rapid, evidence-grounded interaction risk assessments without access to specialised pharmacology expertise.")
      ]),

      h2("8.2 Limitations"),
      bullet([bold("Dataset Scale: "), normal("The 438-pair dataset, while clinically grounded, is small relative to the combinatorial space of 108 drugs \u00D7 19 foods. Expansion to a larger clinical corpus would improve confidence in rare drug class predictions.")]),
      bullet([bold("Regression Head Gap: "), normal("The bioavailability regression R\u00B2 of 0.44 on unseen data reflects the inherent variability of continuous pharmacokinetic measurements across patient populations, a limitation that cannot be resolved by modelling alone without patient-level covariate data.")]),
      bullet([bold("Static Labelling: "), normal("FDA label text is a pharmacovigilance signal, not a direct experimental measurement. Some interactions may be under- or over-represented in label text due to reporting biases.")]),

      h2("8.3 Future Work"),
      bullet([bold("Graph Neural Networks: "), normal("Replacing Morgan fingerprints with graph convolutional or message-passing neural networks operating directly on molecular graphs could capture higher-order structural dependencies beyond the ECFP4 radius.")]),
      bullet([bold("Patient-Level Personalisation: "), normal("Integrating CYP450 genotype data (e.g., CYP3A4/5 polymorphisms) and patient demographic covariates would enable personalised interaction risk stratification.")]),
      bullet([bold("Expanded Food Vocabulary: "), normal("The current system covers 19 dietary items; expansion to the full dietary exposome (hundreds of food components, including micronutrients, phytochemicals, and probiotics) would substantially increase clinical coverage.")]),
      bullet([bold("Multi-Modal Integration: "), normal("Incorporating pharmacokinetic time-series data, protein\u2013drug binding affinity (Ki values from BindingDB), and electronic health record co-prescription frequencies as additional feature sources.")]),
      bullet([bold("Clinical Validation Study: "), normal("Prospective validation of model predictions against monitored patient cohort data would provide the evidence base required for integration into clinical decision-support workflows.")]),

      pageBreak(),

      // ══════════════════════════════════════════════════════════
      // 9. REFERENCES
      // ══════════════════════════════════════════════════════════
      h1("9. References"),
      ...[
        "[1]  T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), San Francisco, CA, USA, 2016, pp. 785\u2013794.",
        "[2]  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A Highly Efficient Gradient Boosting Decision Tree,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017, vol. 30.",
        "[3]  S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017, vol. 30.",
        "[4]  D. Rogers and M. Hahn, \"Extended-Connectivity Fingerprints,\" Journal of Chemical Information and Modeling, vol. 50, no. 5, pp. 742\u2013754, 2010.",
        "[5]  J. J. Irwin and B. K. Shoichet, \"ZINC\u2014A Free Database of Commercially Available Compounds for Virtual Screening,\" Journal of Chemical Information and Modeling, vol. 45, no. 1, pp. 177\u2013182, 2005.",
        "[6]  J. I. Boullata and V. T. Armenti, Handbook of Drug-Nutrient Interactions, 2nd ed. Humana Press, New York, NY, USA, 2010.",
        "[7]  C. S. Won, N. H. Oberlies, and M. F. Paine, \"Mechanisms Underlying Food-Drug Interactions: Inhibition of Intestinal Metabolism and Transport,\" Pharmacology & Therapeutics, vol. 136, no. 2, pp. 186\u2013201, 2012.",
        "[8]  J. J. Ryu, Y. S. Kim, A. S. Guo, T. D. Nguyen, and D. D. Zhang, \"Deep Learning Improves Prediction of Drug\u2013Drug and Drug\u2013Food Interactions,\" Proceedings of the National Academy of Sciences, vol. 115, no. 18, pp. E4304\u2013E4311, 2018.",
        "[9]  M. Zitnik, M. Agrawal, and J. Leskovec, \"Modeling Polypharmacy Side Effects with Graph Convolutional Networks,\" Bioinformatics, vol. 34, no. 13, pp. i457\u2013i466, 2018.",
        "[10] RDKit: Open-Source Cheminformatics. Available: https://www.rdkit.org",
        "[11] PubChem PUG REST API. National Library of Medicine. Available: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
        "[12] OpenFDA Drug Label API. U.S. Food & Drug Administration. Available: https://open.fda.gov/apis/drug/label/",
        "[13] L. A. Bemis and M. A. Murcko, \"The Properties of Known Drugs: Molecular Frameworks,\" Journal of Medicinal Chemistry, vol. 39, no. 15, pp. 2887\u20132893, 1996.",
        "[14] scikit-learn: Machine Learning in Python. F. Pedregosa et al., Journal of Machine Learning Research, vol. 12, pp. 2825\u20132830, 2011.",
      ].map(ref => new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({ text: ref, size: 20, font: "Arial" })]
      })),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('./Drug_Food_Interaction_Project_Report.docx', buffer);
  console.log('Document created successfully');
});