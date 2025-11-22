# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import base64
import os
from datetime import datetime
from dotenv import load_dotenv
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import tempfile
import smtplib
import ssl
from email.message import EmailMessage

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

app = FastAPI(title="Mediredy AI Doctor v2.0", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# SMTP config for emailing PDFs
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)

# API URLs
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ELEVENLABS_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

# Models
OPENAI_VISION_MODEL = "gpt-4o"  # GPT-4 Omni - supports vision

# -------------------------
# Pydantic Models
# -------------------------
class SymptomRequest(BaseModel):
    symptoms: List[str]
    additional_info: Optional[str] = None

class VitalSigns(BaseModel):
    heart_rate: Optional[int] = None
    spo2: Optional[int] = None
    temperature: Optional[float] = None
    blood_pressure: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    history: Optional[List[dict]] = []

class VoiceMessage(BaseModel):
    text: str

class PrescriptionRequest(BaseModel):
    patient_name: str
    patient_age: int
    diagnosis: str
    medications: List[dict]
    doctor_name: Optional[str] = "AI Doctor"  # overridden to Mediready
    symptoms: Optional[List[str]] = []
    advice: Optional[str] = None
    patient_email: Optional[str] = None

# -------------------------
# Helper functions
# -------------------------
def call_openai_chat(messages: List[dict]):
    """Chat using OpenAI API"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",  # use gpt-4o if available
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] OpenAI Chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")

def call_openai_vision(image_data: str, prompt: str):
    """Analyze images with GPT-4o Vision"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    if not image_data.startswith('data:'):
        image_data = f"data:image/jpeg;base64,{image_data}"
    
    payload = {
        "model": OPENAI_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] OpenAI Vision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI Vision Error: {str(e)}")

def text_to_speech_elevenlabs(text: str):
    if not ELEVENLABS_API_KEY:
        return None
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    try:
        response = requests.post(ELEVENLABS_TTS_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[WARN] ElevenLabs TTS failed: {str(e)}")
        return None

# PDF helpers (unchanged)
def _make_prescriptions_dir():
    base = tempfile.gettempdir() or "."
    prescriptions_dir = os.path.join(base, "mediredy_prescriptions")
    os.makedirs(prescriptions_dir, exist_ok=True)
    return prescriptions_dir

def generate_prescription_pdf_file(data: PrescriptionRequest):
    data.doctor_name = "Mediready"
    prescriptions_dir = _make_prescriptions_dir()
    filename = f"prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.pdf"
    filepath = os.path.join(prescriptions_dir, filename)
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "MEDICAL PRESCRIPTION")
    c.line(50, height - 60, width - 50, height - 60)
    # Date
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    # Patient Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 110, "Patient Information")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 130, f"Name: {data.patient_name}")
    c.drawString(50, height - 145, f"Age: {data.patient_age} years")
    y_pos = height - 175
    # Symptoms
    if data.symptoms and len(data.symptoms) > 0:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Presenting Symptoms")
        c.setFont("Helvetica", 10)
        y_pos -= 20
        for symptom in data.symptoms[:10]:
            c.drawString(70, y_pos, f"• {symptom}")
            y_pos -= 15
        y_pos -= 10
    # Diagnosis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Diagnosis")
    c.setFont("Helvetica", 11)
    y_pos -= 20
    diagnosis_lines = simpleSplit(data.diagnosis, "Helvetica", 11, width - 120)
    for line in diagnosis_lines[:20]:
        c.drawString(70, y_pos, line)
        y_pos -= 15
        if y_pos < 120:
            c.showPage()
            y_pos = height - 80
    y_pos -= 10
    # Medications
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Prescribed Medications")
    c.setFont("Helvetica", 10)
    y_pos -= 20
    for i, med in enumerate(data.medications[:20], 1):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, y_pos, f"{i}. {med.get('name', 'Medication')}")
        y_pos -= 15
        c.setFont("Helvetica", 9)
        c.drawString(90, y_pos, f"Dosage: {med.get('dosage', 'As directed')}")
        y_pos -= 12
        c.drawString(90, y_pos, f"Frequency: {med.get('frequency', 'As directed')}")
        y_pos -= 12
        c.drawString(90, y_pos, f"Duration: {med.get('duration', 'As directed')}")
        y_pos -= 18
        if y_pos < 120:
            c.showPage()
            y_pos = height - 80
    # Advice
    if data.advice:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_pos, "Medical Advice")
        c.setFont("Helvetica", 10)
        y_pos -= 20
        advice_lines = simpleSplit(data.advice, "Helvetica", 10, width - 120)
        for line in advice_lines[:50]:
            c.drawString(70, y_pos, line)
            y_pos -= 15
            if y_pos < 120:
                c.showPage()
                y_pos = height - 80
    # Doctor signature
    y_pos -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_pos, f"Prescribing Physician: Dr. {data.doctor_name}")
    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 60, "DISCLAIMER: This is an AI-generated prescription template.")
    c.drawString(50, 48, "Must be reviewed and signed by a licensed medical practitioner before use.")
    c.drawString(50, 36, "Powered by Mediredy AI Doctor v2.0")
    c.save()
    return filepath

# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
async def root():
    return {
        "message": "Mediredy AI Doctor v2.0",
        "version": "2.0.0",
        "features": {
            "vision_analysis": bool(OPENAI_API_KEY),
            "fast_chat": True,
            "natural_voice": bool(ELEVENLABS_API_KEY)
        },
        "models": {
            "vision": OPENAI_VISION_MODEL,
            "chat": "OpenAI GPT-4o",
            "voice": "ElevenLabs" if ELEVENLABS_API_KEY else "Browser"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "apis": {
            "openai": "✅" if OPENAI_API_KEY else "❌",
            "elevenlabs": "✅" if ELEVENLABS_API_KEY else "❌"
        }
    }

@app.get("/test")
async def test():
    return {
        "openai": "configured" if OPENAI_API_KEY else "missing",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "missing"
    }

# Chat
@app.post("/api/chat")
async def chat_with_doctor(message: ChatMessage):
    messages = [
        {"role": "system", "content": """You are Dr. Mediredy, an AI medical doctor.
Be:
- Empathetic and caring
- Clear and conversational
- Concise (100-150 words for voice)
- Recommend professional consultation for serious issues"""}
    ]
    for msg in message.history[-10:]:
        messages.append(msg)
    messages.append({"role": "user", "content": message.message})

    try:
        response_text = call_openai_chat(messages)
        audio_data = text_to_speech_elevenlabs(response_text)
        audio_base64 = base64.b64encode(audio_data).decode('utf-8') if audio_data else None
        return {"response": response_text, "audio": audio_base64, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        print(f"[ERROR] Chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Symptom Assessment
@app.post("/api/symptom-assessment")
async def symptom_assessment(request: SymptomRequest):
    symptoms_text = ", ".join(request.symptoms)
    additional = request.additional_info or "None"
    prompt = f"""You are an experienced medical doctor. A patient presents with:
Symptoms: {symptoms_text}
Additional Info: {additional}
Provide a professional assessment (250 words max):
1. Most likely conditions (2-3 differential diagnoses)
2. Urgency level (Low/Medium/High)
3. Recommended actions
4. When to seek care
5. Warning signs
Be empathetic and clear."""

    messages = [{"role": "system", "content": "You are Dr. Mediredy, an AI medical assistant."},
                {"role": "user", "content": prompt}]
    try:
        assessment = call_openai_chat(messages)
        return {"symptoms": request.symptoms, "assessment": assessment,
                "timestamp": datetime.now().isoformat(),
                "disclaimer": "AI-generated advice. Consult healthcare professionals."}
    except Exception as e:
        print(f"[ERROR] Symptom assessment: {str(e)}")
        raise

# Vital signs
@app.post("/api/vital-signs")
async def vital_signs_analysis(vitals: VitalSigns):
    vital_summary = f"""Heart Rate: {vitals.heart_rate or 'N/A'} bpm
SpO2: {vitals.spo2 or 'N/A'}%
Temperature: {vitals.temperature or 'N/A'}°F
Blood Pressure: {vitals.blood_pressure or 'N/A'}"""

    prompt = f"""Analyze these vital signs:
{vital_summary}
Provide (200 words max):
1. Assessment of each sign
2. Clinical significance
3. Concerns
4. Recommendations"""

    messages = [{"role": "system", "content": "You are analyzing vital signs."},
                {"role": "user", "content": prompt}]
    try:
        analysis = call_openai_chat(messages)
        return {"vitals": vitals.dict(), "analysis": analysis, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        print(f"[ERROR] Vital signs: {str(e)}")
        raise

# Injury Detection
@app.post("/api/injury-detection")
async def injury_detection(file: UploadFile = File(...)):
    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    prompt = """You are an emergency medicine specialist. Analyze this injury photo.
Provide:
1. Injury Type
2. Severity
3. Characteristics
4. First Aid
5. Warning Signs
6. Follow-up"""
    analysis = call_openai_vision(base64_image, prompt)
    return {"filename": file.filename, "analysis": analysis, "analyzed_by": "GPT-4o Vision",
            "timestamp": datetime.now().isoformat(), "disclaimer": "AI analysis. Seek professional evaluation."}

# Report Analysis
@app.post("/api/report-analysis")
async def report_analysis(file: UploadFile = File(...)):
    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    prompt = """You are a professional radiologist. Analyze the report thoroughly and provide:
1. Report Type
2. Key Findings
3. Anatomical Structures
4. Abnormalities
5. Clinical Significance
6. Recommendations
7. Patient Explanation"""
    analysis = call_openai_vision(base64_image, prompt)
    return {"filename": file.filename, "analysis": analysis, "analyzed_by": "GPT-4o Vision",
            "timestamp": datetime.now().isoformat(), "disclaimer": "Have reports reviewed by qualified professionals."}

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("🏥 MEDIREDY AI DOCTOR v2.0")
    print("="*70)
    print(f"✅ OpenAI: {OPENAI_VISION_MODEL}")
    print(f"✅ ElevenLabs: {'Configured' if ELEVENLABS_API_KEY else 'Using Browser TTS'}")
    print("="*70)
    print(f"📍 http://127.0.0.1:8000")
    print(f"📍 http://localhost:8000/docs")
    print("="*70 + "\n")

    # Run server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

