## 🩺 NephroCare AI — Kidney Disease Diagnostic Portal

> An end-to-end AI-powered clinical decision support system that turns messy clinical notes into structured kidney diagnostics — with **Doctor-grade** and **Patient-friendly** reports.

Built with **Streamlit + LangChain + Groq LLM + Scikit-learn**, this project combines **LLM-based medical data extraction**, a **trained ML classifier for CKD prediction**, and a **beautiful dual-audience summary engine** — all wrapped in a premium, animated medical UI.

---

### 🎯 What It Does

**1. 📋 AI Lab Note Screening**
Paste a clinical note (or upload a `.txt / .csv / .json` file) — the system:
- Uses an **LLM with structured Pydantic extraction** to pull out **20 kidney biomarkers** from free-text notes
- Handles **multiple patients in one file** automatically
- Pre-fills the manual calculator for review/edit

**2. 🎛️ Manual Lab Calculator**
A clean form to enter/edit all 20 UCI kidney biomarkers:
`age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc, hemo, pcv, rc, htn, dm, cad, appet, pe, ane`

**3. 🧠 ML Prediction + LLM Summarization**
- Trained classifier (`kidney_model.pkl`) → predicts **CKD vs Healthy** with confidence
- LLM generates **5 structured sections**:
  - Abnormal Factors
  - Top CKD Drivers
  - **Doctor Summary** (clinical, natural tone)
  - **Patient Summary** (1 warm, plain-English paragraph)
  - Recommended Actions

**4. 🗂️ Patient Registry**
- Every screening auto-saved to `patient_reports.json`
- Fully sortable/searchable **HTML registry table** with all 20 biomarkers, flags for abnormal values, and confidence bars
- Export to **CSV** or **JSON**

---

### ✨ Standout Features

| Feature | Description |
|---|---|
| 🎬 **Live AI Pipeline Animation** | Watch data travel through: `Patient Data → LLM → ML Trees → LLM → Report` |
| 🎯 **Confidence Ring** | Radial progress visual for prediction confidence |
| 🧪 **20-Biomarker Grid** | Color-coded normal/abnormal tiles with reference ranges |
| 👨‍⚕️ **Dual-Report Design** | Side-by-side Doctor card + Patient care guide |
| 🔍 **Multi-Patient Extraction** | Handles batch lab reports in a single upload |
| 💾 **Full Audit Trail** | JSON persistence + downloadable printable report (`.md`) |
| 🎨 **Custom Medical UI** | Fully themed Streamlit CSS — rounded cards, glass effects, soft glows |

---

### 🧰 Tech Stack

```
🐍 Python 3.10+
🎈 Streamlit           → Frontend + UI state
🦜 LangChain + Groq    → LLM extraction & summary chains
🤖 Scikit-learn        → CKD classifier (joblib .pkl)
📊 Pandas              → Feature encoding & data prep
✅ Pydantic            → Structured output validation
💾 JSON                → Lightweight persistence layer
```

---

### 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Clinical     │────▶│ LLM Extract  │────▶│ ML Predict   │────▶│ LLM Summary  │
│ Note / File  │     │ (Pydantic)   │     │ (RandomFor.) │     │ (Doctor+Pt)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────┐
                                          │ patient_reports.json │
                                          │  + Registry Table    │
                                          └──────────────────────┘
```

---

### 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/ikrmrind/nephrocare-ai.git
cd nephrocare-ai

# 2. Install
pip install streamlit langchain langchain-groq pydantic pandas scikit-learn joblib python-dotenv

# 3. Add your Groq API key
echo "GROQ_API_KEY=gsk_..." > .env

# 4. Place your trained model
# kidney_model.pkl  (or upload it via the sidebar)

# 5. Run
streamlit run app.py
```

---

### 🎬 Demo Cases Included

- 🔴 **CKD Patient Demo** — Ahmed Raza, 62 (HTN + DM + creatinine 3.8)
- 🟢 **Healthy Patient Demo** — Sara Khan, 34 (all markers normal)

One click → full pipeline animation → dual report → saved record.

---

### 🧪 Why This Matters

> Chronic Kidney Disease affects **~850 million people worldwide**, yet is often caught late because lab data lives in unstructured notes.
> 
> **NephroCare AI** demonstrates how LLMs + classical ML can bridge that gap — turning free-text clinical reports into:
> - ✅ Structured predictions
> - ✅ Actionable doctor summaries
> - ✅ Understandable patient guidance
> - ✅ A clean audit trail

---

### 🛣️ Roadmap

- [ ] PDF lab-report parsing (PyMuPDF)
- [ ] Multi-language patient summaries (Urdu, Hindi, Arabic)
- [ ] SHAP explainability per prediction
- [ ] FHIR / HL7 export for hospital integration
- [ ] Docker deployment + FastAPI wrapper
- [ ] Model retraining dashboard

---

### 🙏 Acknowledgements

- **UCI Machine Learning Repository** — Chronic Kidney Disease dataset
- **Groq** for blazing-fast LLM inference
- **LangChain** for composable chains
- **Streamlit** for rapid clinical UI

---

### ⚠️ Disclaimer

> This is a **research and educational project**, not a certified medical device. Predictions must **never** replace professional clinical judgment. Always consult a nephrologist for real diagnostic decisions.

---

### 👨‍💻 Author

**Muhammad Ikram** — Data Scientist 
🔗 [LinkedIn]https://www.linkedin.com/in/muhammad-ikram-a6b74b31a/ •

---

### ⭐ If this project inspires you, drop a star!

`#HealthcareAI` `#LangChain` `#Groq` `#Streamlit` `#MachineLearning` `#Nephrology` `#LLM` `#MedicalAI` `#Python` `#CKD`
