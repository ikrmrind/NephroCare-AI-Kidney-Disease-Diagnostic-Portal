from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import json
import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Union, Optional

load_dotenv()


llm = ChatGroq(model="openai/gpt-oss-120b")


# Extractor
class Extractor(BaseModel):
    name: Optional[str] = None
    age: Optional[Union[int, float]] = None
    bp: Optional[Union[int, float]] = None
    al: Optional[Union[int, float]] = None
    su: Optional[Union[int, float]] = None
    rbc: Optional[str] = None
    pc: Optional[str] = None
    pcc: Optional[str] = None
    ba: Optional[str] = None
    bgr: Optional[Union[int, float]] = None
    bu: Optional[Union[int, float]] = None
    sc: Optional[Union[int, float]] = None
    hemo: Optional[Union[int, float]] = None
    pcv: Optional[Union[int, float]] = None
    rc: Optional[Union[int, float]] = None
    htn: Optional[str] = None
    dm: Optional[str] = None
    cad: Optional[str] = None
    appet: Optional[str] = None
    pe: Optional[str] = None
    ane: Optional[str] = None


# Config
MODEL_PATH = r"D:\RAG\InnovateX\kidney_model.pkl"
JSON_PATH = "patient_reports.json"

FEATURES = [
    "age", "bp", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
    "sc", "hemo", "pcv", "rc", "htn", "dm", "cad", "appet", "pe", "ane",
]

CATEGORICAL_MAP = {
    "rbc":   {"normal": 0, "abnormal": 1},
    "pc":    {"normal": 0, "abnormal": 1},
    "pcc":   {"notpresent": 0, "present": 1},
    "ba":    {"notpresent": 0, "present": 1},
    "htn":   {"no": 0, "yes": 1},
    "dm":    {"no": 0, "yes": 1},
    "cad":   {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe":    {"no": 0, "yes": 1},
    "ane":   {"no": 0, "yes": 1},
}

DEFAULTS = {
    "age": 50, "bp": 80, "al": 0, "su": 0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "bgr": 120, "bu": 40, "sc": 1.2, "hemo": 13.0,
    "pcv": 40, "rc": 5.0,
    "htn": "no", "dm": "no", "cad": "no",
    "appet": "good", "pe": "no", "ane": "no",
}

ml_model = joblib.load(MODEL_PATH)


# Predict

def predict(user_data: dict):
    """Copy → encode → predict. Original not modified."""
    data = user_data.copy()

    for k, v in DEFAULTS.items():
        if data.get(k) is None:
            data[k] = v

    for col, mapping in CATEGORICAL_MAP.items():
        val = str(data[col]).strip().lower()
        data[col] = mapping.get(val, 0)

    X = pd.DataFrame([data])[FEATURES].values

    pred = int(ml_model.predict(X)[0])
    proba = ml_model.predict_proba(X)[0]
    confidence = float(proba.max() * 100)

    return pred, confidence


# Save record (with summary)
def save_record(user_data: dict, pred: int, confidence: float, summary: str = ""):
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = [records]
    else:
        records = []

    entry = {
        "id": len(records) + 1,
        "timestamp": datetime.now().isoformat(),
        "user_data": user_data.copy(),
        "prediction": {
            "label": "CKD" if pred == 1 else "Healthy",
            "value": pred,
            "confidence": round(confidence, 2),
        },
        "summary": summary,
    }
    records.append(entry)

    with open(JSON_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved record #{entry['id']} → {JSON_PATH}")
    return entry


# LLM extractor chain
parser = PydanticOutputParser(pydantic_object=Extractor)

template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a medical laboratory data extraction assistant for a kidney disease project.
Extract structured patient information from the text below.

Return values EXACTLY as follows:
- htn, dm, cad, pe, ane : "yes" or "no"
- rbc, pc               : "normal" or "abnormal"
- pcc, ba               : "present" or "notpresent"
- appet                 : "good" or "poor"
- numeric fields         : numbers only (no units)
- If a value is missing, leave it null.

{format_instructions}"""),
    ("user", "{paragraph}"),
])

chain = template | llm | parser


def extract_from_text(text: str) -> dict:
    result = chain.invoke({
        "paragraph": text,
        "format_instructions": parser.get_format_instructions(),
    })
    return result.model_dump()


# Summary prompt & chain
Summary_Temp = ChatPromptTemplate.from_messages([
    ("system",
     """You are a kidney disease analyst. Output a SHORT summary — max 120 words total.

Format STRICTLY:

**Abnormal Factors:** bullet list (max 4, only abnormal values)
**Top CKD Drivers:** one line (max 3 causes)
**Doctor:** 1–2 sentences (technical)
**Patient:** 1–2 sentences (plain language)
**Actions:** 2–3 short bullets

Rules:
- Only mention abnormal values (sc > 1.5, bu > 40, hemo < 12, al ≥ 1, htn=yes, dm=yes, etc.)
- No greetings, no disclaimers, no repetition
- Be blunt and concise"""),
    ("user",
     """Data: {patient_data}
Prediction: {prediction_label} ({confidence}%)

Summarize now.""")
])

summary_chain = Summary_Temp | llm


def generate_summary(user_data: dict, pred: int, confidence: float) -> str:
    label = "CKD (Chronic Kidney Disease)" if pred == 1 else "Healthy (No CKD)"

    response = summary_chain.invoke({
        "patient_data": json.dumps(user_data, indent=2),
        "prediction_label": label,
        "confidence": round(confidence, 2),
    })
    return response.content


# Output pipeline (predict + summary + save + print)
def output_process(user_data: dict):
    data_copy = user_data.copy()
    pred, confidence = predict(data_copy)

    print("🧠 Generating medical summary...")
    summary = generate_summary(user_data, pred, confidence)

    entry = save_record(user_data, pred, confidence, summary)

    print("\n" + "=" * 60)
    print(f"Record #     : {entry['id']}")
    print(f"Prediction   : {'⚠️  CKD' if pred == 1 else '✅  Healthy'}")
    print(f"Confidence   : {confidence:.2f}%")
    print("-" * 60)
    print("📋 Summary for Doctor & Patient:\n")
    print(summary)
    print("=" * 60)

    return entry


# Menu (runs LAST)
print(" (1)  Past Information : ")
print(" (2)  Upload File : ")

select = int(input("Select Your Process : "))

if select == 1:
    user_text = input("Past Data : ")
    extracted = extract_from_text(user_text)
    print("\n🔍 Extracted:")
    print(json.dumps(extracted, indent=2))
    output_process(extracted)

elif select == 2:
    file_path = input("File path : ").strip().strip('"')
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    extracted = extract_from_text(text)
    print("\n🔍 Extracted:")
    print(json.dumps(extracted, indent=2))
    output_process(extracted)