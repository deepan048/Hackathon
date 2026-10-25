"""
Brand Shield Guardian - Backend Engine
Evidence-based Brand Governance, FAISS Retrieval, PDF Parsing & Compliance API

Run:  uvicorn backend:app --reload --port 8000
NOTE: this file must NOT import or call streamlit.
"""

import os
import re
import json
import base64
import asyncio
import logging
import hashlib
from typing import List, Optional, Dict, Any

import numpy as np
from dotenv import load_dotenv

# PDF processing
import pymupdf  # PyMuPDF

# FastAPI & Pydantic
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain, FAISS & embeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brand_shield_guardian")

app = FastAPI(  # uvicorn looks for this name: backend:app
    title="Brand Shield Guardian API",
    description="Evidence-based marketing governance and compliance engine",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Fallback Deterministic Embeddings -----------------
class DeterministicLocalEmbeddings(Embeddings):
    """Offline hashed bag-of-words embeddings so FAISS always works."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def _has_openai_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    return key if key and not key.startswith("your-") else None


def get_embeddings_model():
    """OpenAI embeddings if configured, otherwise the local deterministic ones."""
    api_key = _has_openai_key()
    if api_key:
        try:
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(openai_api_key=api_key)
        except Exception as e:
            logger.warning(f"OpenAIEmbeddings unavailable ({e}). Using DeterministicLocalEmbeddings.")
    return DeterministicLocalEmbeddings()


# ----------------- Data Models -----------------
class ReviewRequest(BaseModel):
    content: str
    guidelines: str


class Issue(BaseModel):
    issue_type: str = Field(..., description="forbidden_term | factual_contradiction | unsupported_claim | tone_violation | visual_mismatch")
    severity: str = Field(..., description="high | medium | low")
    detected_text: str
    rule_violated: str
    explanation: str
    evidence: str
    suggested_fix: str
    confidence: float = 0.95


class SeverityBreakdown(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0


class ReviewResponse(BaseModel):
    total_violations: int
    compliance_status: str  # "Pass" | "Requires Revision" | "Critical Violation"
    overall_status: str     # "PASS" | "FAIL"
    severity_breakdown: SeverityBreakdown
    issues: List[Issue] = []
    corrected_text: str = ""
    retrieved_evidence: List[str] = []
    extracted_text: Optional[str] = ""


# ----------------- FAISS Retrieval -----------------
def retrieve_relevant_guidelines(guidelines_text: str, query_content: str, top_k: int = 4) -> List[str]:
    """Chunk guidelines, index in FAISS, retrieve the rules most relevant to the content."""
    if not guidelines_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "Rule ", "Section ", "; ", ". "],
    )
    docs = splitter.create_documents([guidelines_text])
    if not docs:
        return [guidelines_text]

    try:
        vs = FAISS.from_documents(docs, get_embeddings_model())
        query = query_content[:500] if query_content else guidelines_text[:200]
        results = vs.similarity_search(query, k=min(top_k, len(docs)))
        return [d.page_content.strip() for d in results if d.page_content.strip()]
    except Exception as e:
        logger.warning(f"FAISS retrieval fallback: {e}")
        return [d.page_content.strip() for d in docs[:top_k]]


# ----------------- File Extraction -----------------
def extract_text_from_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """PyMuPDF text extraction; scanned PDFs get page 1 rendered for vision review."""
    extracted_text = ""
    rendered_image_b64 = None
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    extracted_text += page_text + "\n"

            if not extracted_text.strip() and page_count > 0:
                pix = doc[0].get_pixmap(dpi=150)
                rendered_image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                extracted_text = "[Scanned / Image-Based PDF: first page converted for multimodal vision review]"

        return {"text": extracted_text.strip(), "rendered_image_b64": rendered_image_b64, "page_count": page_count}
    except Exception as e:
        logger.error(f"PyMuPDF error: {e}")
        return {"text": "", "rendered_image_b64": None, "page_count": 0}


# ----------------- Result normalisation -----------------
def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Make LLM/heuristic output match ReviewResponse and keep counts/status consistent."""
    issues = []
    for raw in data.get("issues") or []:
        sev = str(raw.get("severity", "medium")).lower()
        if sev == "critical":
            sev = "high"
        if sev not in ("high", "medium", "low"):
            sev = "medium"
        try:
            conf = float(raw.get("confidence", 0.9))
        except (TypeError, ValueError):
            conf = 0.9
        issues.append({
            "issue_type": str(raw.get("issue_type") or "unsupported_claim"),
            "severity": sev,
            "detected_text": str(raw.get("detected_text") or "N/A"),
            "rule_violated": str(raw.get("rule_violated") or "Brand guideline"),
            "explanation": str(raw.get("explanation") or ""),
            "evidence": str(raw.get("evidence") or ""),
            "suggested_fix": str(raw.get("suggested_fix") or ""),
            "confidence": conf,
        })

    high = sum(i["severity"] == "high" for i in issues)
    med = sum(i["severity"] == "medium" for i in issues)
    low = sum(i["severity"] == "low" for i in issues)

    if high:
        compliance_status, overall_status = "Critical Violation", "FAIL"
    elif med or low:
        compliance_status, overall_status = "Requires Revision", "FAIL"
    else:
        compliance_status, overall_status = "Pass", "PASS"

    return {
        "total_violations": len(issues),
        "compliance_status": compliance_status,
        "overall_status": overall_status,
        "severity_breakdown": {"high": high, "medium": med, "low": low},
        "issues": issues,
        "corrected_text": str(data.get("corrected_text") or ""),
        "retrieved_evidence": [str(r) for r in (data.get("retrieved_evidence") or [])],
    }


# ----------------- Heuristic Local Audit -----------------
TIME_CLAIM_RE = re.compile(
    r"(guaranteed\s+)?(results?\s+)?(in\s+|within\s+)?(just\s+)?(an?\s+)?"
    r"(instant(?:ly)?|1[- ]day|one[- ]day|overnight|24[- ]hours?)(\s+results?)?",
    re.IGNORECASE,
)
URGENCY_RE = re.compile(r"\b(order now|buy now|before stock runs out|hurry)\b[^.!?]*", re.IGNORECASE)

PROHIBITED_TERMS = {
    "miracle": "clinically formulated",
    "overnight results": "gradual results",
    "complete elimination": "visible reduction",
    "100% cure": "targeted care",
    "guaranteed to transform": "formulated to support",
}


def extract_approved_names(guidelines: str) -> List[str]:
    """Quoted product names found on nomenclature / product-name rule lines."""
    names: List[str] = []
    for line in guidelines.splitlines():
        if re.search(r"nomenclature|product name|refer to (?:the )?product", line, re.IGNORECASE):
            for n in re.findall(r"['\"]([A-Z][\w ]*?)['\"]", line):
                if n not in names:
                    names.append(n)
    return names


def audit_locally_heuristic(content: str, guidelines: str, retrieved_rules: List[str]) -> Dict[str, Any]:
    """Deterministic rule-matching audit used when no OpenAI key is available."""
    issues: List[Dict[str, Any]] = []
    content_lower = content.lower()
    guidelines_lower = guidelines.lower()
    corrected = content
    time_issue_raised = False

    # 1. Instant / 1-day / overnight claims
    m = TIME_CLAIM_RE.search(content)
    if m and any(t in guidelines_lower for t in ["instant", "1-day", "1 day", "overnight"]):
        detected = m.group(0).strip()
        fix = "visible improvement over 4 weeks of regular use"
        issues.append({
            "issue_type": "factual_contradiction",
            "severity": "high",
            "detected_text": detected,
            "rule_violated": "Rule: Do not use instant, 1-day, or overnight guarantees.",
            "explanation": "Guarantees of instant or overnight results contradict verified clinical trial timelines.",
            "evidence": "Approved claim: 'Results show improvement over 4 weeks of regular use.'",
            "suggested_fix": fix,
            "confidence": 0.98,
        })
        corrected = TIME_CLAIM_RE.sub(fix, corrected, count=1)
        time_issue_raised = True

    # 2. Prohibited terms
    for term, fix in PROHIBITED_TERMS.items():
        if term not in content_lower:
            continue
        if term == "overnight results" and time_issue_raised:
            continue  # already covered by the time-claim issue
        match = re.search(re.escape(term), content, re.IGNORECASE)
        detected = match.group(0) if match else term
        issues.append({
            "issue_type": "forbidden_term",
            "severity": "high",
            "detected_text": detected,
            "rule_violated": f"Rule: Prohibited term '{term.title()}' is strictly forbidden.",
            "explanation": f"The term '{detected}' is blacklisted in the brand terminology policy.",
            "evidence": f"Guidelines prohibit hyperbolic terms like '{term}'.",
            "suggested_fix": fix,
            "confidence": 0.99,
        })
        corrected = re.sub(re.escape(term), fix, corrected, flags=re.IGNORECASE)

    # 3. Product nomenclature
    approved_names = extract_approved_names(guidelines)
    if approved_names and not any(n.lower() in content_lower for n in approved_names):
        issues.append({
            "issue_type": "forbidden_term",
            "severity": "medium",
            "detected_text": "Missing official product nomenclature",
            "rule_violated": f"Rule: Product name must follow official nomenclature: {', '.join(approved_names)}.",
            "explanation": f"The copy does not reference the official brand name ({approved_names[0]}).",
            "evidence": f"Approved nomenclature required: '{approved_names[0]}'.",
            "suggested_fix": f"Reference the official product as '{approved_names[0]}'.",
            "confidence": 0.92,
        })
        corrected = f"{approved_names[0]}: {corrected}"

    # 4. High-pressure urgency
    m = URGENCY_RE.search(content)
    if m:
        detected = m.group(0).strip()
        fix = "Discover how our formula supports your routine today"
        issues.append({
            "issue_type": "tone_violation",
            "severity": "medium",
            "detected_text": detected,
            "rule_violated": "Rule: Maintain professional brand tone; avoid high-pressure sales jargon.",
            "explanation": "Urgency language contradicts the professional, scientific brand voice.",
            "evidence": "Brand tone policy mandates professional, encouraging communication.",
            "suggested_fix": fix,
            "confidence": 0.91,
        })
        corrected = URGENCY_RE.sub(fix, corrected, count=1)

    result = normalize_result({
        "issues": issues,
        "corrected_text": corrected.strip(),
        "retrieved_evidence": retrieved_rules,
    })
    return result


# ----------------- LLM Core Audit Engine -----------------
SYSTEM_PROMPT = """
You are Brand Shield Guardian, an expert evidence-based brand governance and compliance auditor.
Evaluate marketing copy or visual ads strictly against the approved brand guidelines and reference facts.

CORE RULES:
1. RETRIEVE FIRST, COMPARE SECOND: Use only the provided guidelines. Never invent brand policies.
2. COUNT & CATEGORIZE: Identify every violation snippet and categorize severity (high, medium, low).
3. ACTIONABLE REWRITES: Give a compliant suggested fix per issue, and a full 'corrected_text' that is
   publishable and preserves the original creator intent.

OUTPUT SCHEMA (valid JSON only):
{
  "total_violations": <int>,
  "compliance_status": "Pass | Requires Revision | Critical Violation",
  "overall_status": "PASS | FAIL",
  "severity_breakdown": {"high": <int>, "medium": <int>, "low": <int>},
  "issues": [
    {
      "issue_type": "forbidden_term | factual_contradiction | unsupported_claim | tone_violation | visual_mismatch",
      "severity": "high | medium | low",
      "detected_text": "Exact snippet from submitted content",
      "rule_violated": "Specific rule citation from guidelines",
      "explanation": "Why this snippet violates the guideline",
      "evidence": "Supporting guideline text or reference fact",
      "suggested_fix": "Compliant wording alternative",
      "confidence": 0.95
    }
  ],
  "corrected_text": "Complete compliant version of the marketing text with all fixes applied"
}

STATUS CRITERIA:
- Critical Violation (FAIL): prohibited terms, illegal claims, or instant/overnight guarantees.
- Requires Revision (FAIL): moderate tone issues, missing nomenclature, or unsupported minor claims.
- Pass (PASS): zero violations.
"""


def _call_openai(api_key: str, messages: list) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


async def audit_governance(
    content: str,
    guidelines: str,
    image_bytes: Optional[bytes] = None,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """FAISS retrieval + GPT-4o-mini audit (text or image), with heuristic fallback for text."""
    retrieved_rules = await asyncio.to_thread(retrieve_relevant_guidelines, guidelines, content, 4)
    retrieved_context = "\n".join(f"- {r}" for r in retrieved_rules) if retrieved_rules else guidelines

    api_key = _has_openai_key()
    if not api_key:
        if image_bytes:
            raise HTTPException(
                status_code=503,
                detail="Image / scanned-PDF review needs OPENAI_API_KEY (vision). Set it in .env or submit text.",
            )
        logger.info("OPENAI_API_KEY not configured. Running local heuristic audit.")
        return audit_locally_heuristic(content, guidelines, retrieved_rules)

    user_text = f"""
[RELEVANT BRAND GUIDELINES RETRIEVED BY FAISS]
{retrieved_context}

[FULL GUIDELINES REFERENCE]
{guidelines}

[SUBMITTED MARKETING CONTENT]
{content if content else "[Visual Creative attached for evaluation]"}
"""
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type or 'image/png'};base64,{b64_img}"}},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_text})

    try:
        data = await asyncio.to_thread(_call_openai, api_key, messages)
        data["retrieved_evidence"] = retrieved_rules
        return normalize_result(data)
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}")
        if image_bytes:
            raise HTTPException(status_code=502, detail=f"Vision review failed: {e}")
        return audit_locally_heuristic(content, guidelines, retrieved_rules)


# ----------------- API Endpoints -----------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Brand Shield Guardian API",
        "has_openai_key": bool(_has_openai_key()),
    }


@app.post("/reviews", response_model=ReviewResponse)
async def review_text(request: ReviewRequest):
    if not request.guidelines.strip():
        raise HTTPException(status_code=400, detail="Brand guidelines cannot be empty.")
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Marketing content cannot be empty.")

    result = await audit_governance(content=request.content, guidelines=request.guidelines)
    result["extracted_text"] = request.content
    return result


@app.post("/reviews/upload", response_model=ReviewResponse)
async def review_upload(guidelines: str = Form(...), file: UploadFile = File(...)):
    if not guidelines.strip():
        raise HTTPException(status_code=400, detail="Brand guidelines cannot be empty.")

    filename = (file.filename or "").lower()
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extracted_text = ""
    image_bytes = None
    mime_type = file.content_type or "image/png"

    if filename.endswith(".pdf") or "pdf" in mime_type:
        pdf_res = extract_text_from_pdf(file_bytes)
        if pdf_res["page_count"] == 0:
            raise HTTPException(status_code=400, detail="Could not read this PDF.")
        extracted_text = pdf_res["text"]
        if pdf_res["rendered_image_b64"]:
            image_bytes = base64.b64decode(pdf_res["rendered_image_b64"])
            mime_type = "image/png"
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")) or "image" in mime_type:
        image_bytes = file_bytes
        extracted_text = f"[Visual Creative Ad '{file.filename}' processed via Multimodal Vision]"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, PNG, or JPG.")

    result = await audit_governance(
        content=extracted_text, guidelines=guidelines, image_bytes=image_bytes, mime_type=mime_type
    )
    result["extracted_text"] = extracted_text
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)