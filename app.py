"""
Brand Shield Guardian - Streamlit UI
Run:  streamlit run app.py
Talks to the FastAPI backend (uvicorn backend:app --port 8000); falls back to the
in-process heuristic engine from backend.py if the API is not running.
"""

import html
import json
from typing import Any, Dict, Optional

import requests
import streamlit as st

st.set_page_config(page_title="Brand Shield Guardian", page_icon="🛡️", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
.brand-header{padding:1.2rem 1.6rem;border-radius:14px;margin-bottom:1rem;
  background:linear-gradient(135deg,#1e1b4b,#312e81);border:1px solid #4338ca}
.brand-title{font-size:2rem;font-weight:800;color:#f8fafc}
.brand-subtitle{margin:.3rem 0 0 0;color:#c7d2fe}
.metric-card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:1rem;text-align:center}
.metric-label{font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8}
.metric-value{font-size:2rem;font-weight:800;margin-top:.2rem}
.panel-card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:1rem 1.2rem;margin-bottom:.8rem}
.status-banner-pass,.status-banner-fail,.status-banner-review{border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem}
.status-banner-pass{background:rgba(16,185,129,.12);border:1px solid #10b981}
.status-banner-fail{background:rgba(239,68,68,.12);border:1px solid #ef4444}
.status-banner-review{background:rgba(245,158,11,.12);border:1px solid #f59e0b}
.badge-critical,.badge-revision,.badge-advisory{padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:700}
.badge-critical{background:#7f1d1d;color:#fecaca}
.badge-revision{background:#78350f;color:#fde68a}
.badge-advisory{background:#1e3a8a;color:#bfdbfe}
.violation-text{background:rgba(239,68,68,.18);color:#fca5a5;padding:2px 6px;border-radius:6px}
.fix-text{background:rgba(16,185,129,.1);border-left:4px solid #10b981;padding:.7rem 1rem;border-radius:8px;margin-top:.5rem}
</style>
""", unsafe_allow_html=True)

DEFAULT_GUIDELINES = (
    "Rule 1: Do not use instant, 1-day, or overnight guarantees. Approved claim: 'Results show improvement over 4 weeks.'\n"
    "Rule 2: Never use prohibited terms such as 'Miracle', 'Overnight Results', or 'Complete Elimination'.\n"
    "Rule 3: Pricing must specify billing interval and currency clearly.\n"
    "Rule 4: Product name must strictly follow official nomenclature 'Lumina Glow' or 'Lumina Health Tech'."
)
DEFAULT_AD = "Get guaranteed results in 1 day with our miracle product! Order now before stock runs out."

st.session_state.setdefault("guidelines_input", DEFAULT_GUIDELINES)
st.session_state.setdefault("ad_input", DEFAULT_AD)


# ----------------- Helpers -----------------
@st.cache_data(ttl=5, show_spinner=False)
def check_backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=1.0).status_code == 200
    except requests.RequestException:
        return False


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value)).replace("\n", "<br/>")


def html_block(markup: str) -> None:
    """Render HTML without indentation turning it into a markdown code block."""
    st.markdown("\n".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def pct(value: Any, default: int = 95) -> int:
    try:
        v = float(value)
        return int(v * 100) if v <= 1 else int(v)
    except (TypeError, ValueError):
        return default


def execute_review(guidelines: str, content: str = "",
                   file_data: Optional[tuple] = None) -> Dict[str, Any]:
    """Send audit request to the FastAPI backend, or fall back to the in-process engine."""
    if check_backend_alive():
        try:
            if file_data:
                filename, file_bytes, mime_type = file_data
                files = {"file": (filename, file_bytes, mime_type or "application/octet-stream")}
                res = requests.post(f"{BACKEND_URL}/reviews/upload",
                                    data={"guidelines": guidelines}, files=files, timeout=90)
            else:
                res = requests.post(f"{BACKEND_URL}/reviews",
                                    json={"content": content, "guidelines": guidelines}, timeout=90)
        except requests.RequestException as e:
            return {"error": f"Lost connection to backend at {BACKEND_URL}: {e}"}

        if res.status_code == 200:
            try:
                return res.json()
            except ValueError:
                return {"error": "Backend returned a non-JSON response."}
        try:
            err = res.json().get("detail", res.text)
        except ValueError:
            err = res.text
        return {"error": f"Backend Error (HTTP {res.status_code}): {err}"}

    # Standalone fallback: import the engine functions from backend.py
    try:
        import backend as local_backend

        if file_data:
            filename, file_bytes, _ = file_data
            if not filename.lower().endswith(".pdf"):
                return {"error": "Image audits need the FastAPI backend (vision). "
                                 "Run `uvicorn backend:app --reload` or upload a PDF / paste text."}
            pdf_res = local_backend.extract_text_from_pdf(file_bytes)
            if pdf_res.get("rendered_image_b64"):
                return {"error": "This PDF is scanned (no text layer). Start the FastAPI backend with OPENAI_API_KEY for vision review."}
            if not pdf_res["text"]:
                return {"error": "Could not read any text from this PDF."}
            extracted_text = pdf_res["text"]
        else:
            extracted_text = content

        res = local_backend.audit_locally_heuristic(extracted_text, guidelines, [])
        res["extracted_text"] = extracted_text
        return res
    except Exception as e:
        return {"error": f"Could not reach backend at {BACKEND_URL} and local fallback failed: {e}"}


# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚙️ Governance Engine")
    if check_backend_alive():
        st.success("🟢 FastAPI Backend: **Connected** (Port 8000)")
    else:
        st.info("🟡 Engine Mode: **Standalone Local Engine**")
        st.caption("Tip: Run `uvicorn backend:app --reload` to activate API mode.")
    st.divider()

    st.markdown("### 📋 Audit Policy Checklist")
    for item in ["Evidence-First Grounding (FAISS)", "Prohibited Claims Detection",
                 "Product Nomenclature Check", "Preserved Creator Intent Rewrites",
                 "Multimodal Vision OCR"]:
        st.markdown(f"- [x] {item}")
    st.divider()

    st.markdown("### 💡 Quick Templates")
    if st.button("Load Cosmetics Ad Sample"):
        st.session_state["guidelines_input"] = DEFAULT_GUIDELINES
        st.session_state["ad_input"] = DEFAULT_AD
    if st.button("Load Fully Compliant Sample"):
        st.session_state["guidelines_input"] = (
            "Rule 1: Results show improvement over 4 weeks of regular use.\n"
            "Rule 2: Avoid aggressive sales jargon. Maintain scientific and professional tone.\n"
            "Rule 3: Always refer to product as 'Lumina Glow'."
        )
        st.session_state["ad_input"] = (
            "Lumina Glow: Clinical trials demonstrate visible improvement in skin hydration "
            "over 4 weeks of regular use. Learn more today."
        )

# ----------------- Header -----------------
html_block("""
<div class="brand-header">
    <div class="brand-title">🛡️ Brand Shield Guardian</div>
    <p class="brand-subtitle">AI-Powered Pre-Publication Brand Governance, FAISS Evidence Grounding &amp; Compliance Audit</p>
</div>
""")

# ----------------- Inputs -----------------
col_g, col_a = st.columns([1, 1], gap="medium")

with col_g:
    st.markdown("#### 1️⃣ Brand Guidelines & Approved Reference Facts")
    guidelines_input = st.text_area(
        "Guidelines Input", key="guidelines_input", height=170, label_visibility="collapsed",
        placeholder="Enter official brand rules, forbidden terms, approved clinical claims, and nomenclature...",
    )

with col_a:
    st.markdown("#### 2️⃣ Marketing Ad Creative to Audit")
    input_type = st.radio("Ad Source", ["Type / Paste Text", "Upload PDF or Image Ad"],
                          horizontal=True, label_visibility="collapsed")
    ad_content_input = ""
    uploaded_file = None
    if input_type == "Type / Paste Text":
        ad_content_input = st.text_area(
            "Ad Text Input", key="ad_input", height=130, label_visibility="collapsed",
            placeholder="Paste your ad copy, social post, email draft, or headline here...",
        )
    else:
        uploaded_file = st.file_uploader("Upload Ad Creative", type=["pdf", "png", "jpg", "jpeg"],
                                         label_visibility="collapsed")
        if uploaded_file and uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            st.image(uploaded_file, caption="Uploaded Creative Preview", width=260)

run_audit = st.button("🚀 Run Brand Compliance Audit", type="primary", width="stretch")

# ----------------- Run -----------------
if run_audit:
    if not guidelines_input.strip():
        st.warning("⚠️ Please provide brand guidelines before running the audit.")
    elif input_type == "Type / Paste Text" and not ad_content_input.strip():
        st.warning("⚠️ Please enter marketing ad copy to audit.")
    elif input_type != "Type / Paste Text" and not uploaded_file:
        st.warning("⚠️ Please select an ad file (PDF, PNG, JPG) to upload.")
    else:
        st.session_state.pop("audit_result", None)  # never show stale results
        with st.spinner("🔍 Grounding against FAISS vector rules & performing compliance audit..."):
            file_data = None
            if input_type != "Type / Paste Text" and uploaded_file:
                file_data = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            result = execute_review(guidelines=guidelines_input,
                                    content=ad_content_input, file_data=file_data)

        if "error" in result:
            st.error(f"❌ **Audit Failed:** {result['error']}")
        else:
            st.session_state["audit_result"] = result

# ----------------- Results -----------------
if "audit_result" in st.session_state:
    data = st.session_state["audit_result"]
    issues = data.get("issues") or []
    total_violations = data.get("total_violations", len(issues))

    sev_of = lambda i: str(i.get("severity", "medium")).lower()
    breakdown = data.get("severity_breakdown") or {}
    high_count = breakdown.get("high", sum(sev_of(i) in ("high", "critical") for i in issues))
    med_count = breakdown.get("medium", sum(sev_of(i) == "medium" for i in issues))
    low_count = breakdown.get("low", sum(sev_of(i) == "low" for i in issues))

    compliance_status = str(data.get("compliance_status") or "Requires Revision")
    overall_status = str(data.get("overall_status") or "FAIL").upper()
    corrected_text = data.get("corrected_text", "")
    extracted_text = data.get("extracted_text", "")
    retrieved_evidence = data.get("retrieved_evidence") or []

    st.divider()

    # 1. Status banner
    if "PASS" in overall_status and total_violations == 0:
        html_block("""
        <div class="status-banner-pass">
            <h3 style="margin:0;color:#10b981;">🟢 AUDIT RESULT: FULLY COMPLIANT</h3>
            <p style="margin:4px 0 0 0;color:#cbd5e1;">Zero brand violations detected. Content adheres to all retrieved guidelines and approved claims.</p>
        </div>""")
    elif "CRITICAL" in compliance_status.upper() or high_count > 0:
        html_block(f"""
        <div class="status-banner-fail">
            <h3 style="margin:0;color:#ef4444;">🔴 AUDIT RESULT: CRITICAL VIOLATION DETECTED</h3>
            <p style="margin:4px 0 0 0;color:#cbd5e1;">Severe brand or legal violations detected ({high_count} Critical). Content must not be published without resolution.</p>
        </div>""")
    else:
        html_block("""
        <div class="status-banner-review">
            <h3 style="margin:0;color:#f59e0b;">🟡 AUDIT RESULT: REQUIRES REVISION</h3>
            <p style="margin:4px 0 0 0;color:#cbd5e1;">Deviations from approved tone, nomenclature, or claims detected. Review suggested corrections below.</p>
        </div>""")

    # 2. Metrics
    def metric_card(label: str, value: Any, color: str, size: str = "") -> str:
        style = f"color:{color};" + (f"font-size:{size};" if size else "")
        return (f'<div class="metric-card"><div class="metric-label">{esc(label)}</div>'
                f'<div class="metric-value" style="{style}">{esc(value)}</div></div>')

    ok, bad, warn = "#10b981", "#ef4444", "#f59e0b"
    cards = [
        ("Total Violations", total_violations, ok if total_violations == 0 else bad, ""),
        ("Compliance Status", compliance_status,
         ok if total_violations == 0 else (bad if high_count > 0 else warn), "1.15rem"),
        ("Critical / High", high_count, ok if high_count == 0 else bad, ""),
        ("Moderate / Advisory", med_count + low_count,
         ok if (med_count + low_count) == 0 else warn, ""),
    ]
    for col, (label, value, color, size) in zip(st.columns(4), cards):
        with col:
            html_block(metric_card(label, value, color, size))

    # 3. Rewrite
    if corrected_text:
        st.markdown("### ✨ Compliant Rewritten Copy (Creator Intent Preserved)")
        html_block(f"""
        <div class="panel-card" style="border-left:4px solid #6366f1;">
            <p style="margin:0;font-size:1.05rem;line-height:1.6;color:#f8fafc;">{esc(corrected_text)}</p>
        </div>""")
        col_c1, col_c2 = st.columns([1, 4])
        with col_c1:
            st.download_button("📥 Download Audit (JSON)",
                               data=json.dumps(data, indent=2, default=str),
                               file_name="brand_guardian_audit.json",
                               mime="application/json", width="stretch")
        with col_c2:
            with st.expander("📋 View Copy-Ready Text Code"):
                st.code(corrected_text, language="text")

    # 4. Violations
    st.markdown("### 🔍 Granular Rule Violations & Evidence Citations")
    if not issues:
        st.info("🎉 No rule violations or prohibited terms detected in the submitted content!")
    else:
        for idx, issue in enumerate(issues):
            sev = sev_of(issue)
            if sev in ("high", "critical"):
                badge_class, badge_label = "badge-critical", "CRITICAL VIOLATION"
            elif sev == "medium":
                badge_class, badge_label = "badge-revision", "REQUIRES REVISION"
            else:
                badge_class, badge_label = "badge-advisory", "ADVISORY"

            with st.expander(f"Violation #{idx + 1}: {issue.get('rule_violated', 'Rule Violation')}",
                             expanded=True):
                col_i1, col_i2 = st.columns([3, 1])
                with col_i1:
                    html_block(f"<span class='{badge_class}'>{badge_label}</span> &nbsp; "
                               f"<b>Category:</b> <code>{esc(issue.get('issue_type', 'General'))}</code>")
                with col_i2:
                    st.caption(f"Confidence: {pct(issue.get('confidence', 0.95))}%")

                html_block(f"<b>Non-Compliant Text Found:</b> "
                           f"<span class='violation-text'>{esc(issue.get('detected_text', 'N/A'))}</span>")
                st.markdown(f"**Explanation:** {issue.get('explanation', 'N/A')}")
                st.markdown(f"**Supporting Evidence:** _{issue.get('evidence', 'N/A')}_")
                html_block(f"""
                <div class="fix-text"><b>Suggested Compliant Reword:</b><br/>
                {esc(issue.get('suggested_fix', 'No fix suggested'))}</div>""")

                col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 3])
                with col_btn1:
                    if st.button("✅ Accept Fix", key=f"accept_{idx}"):
                        st.toast(f"Accepted fix for Violation #{idx + 1}!")
                with col_btn2:
                    if st.button("🛡️ Override / Dismiss", key=f"dismiss_{idx}"):
                        st.toast(f"Violation #{idx + 1} overridden by brand reviewer.")

    # 5. Debug panel
    with st.expander("🔍 View FAISS Retrieved Rules & Extracted Content"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**FAISS Grounded Evidence Rules:**")
            if retrieved_evidence:
                for r in retrieved_evidence:
                    st.markdown(f"- `{r}`")
            else:
                st.caption("All guideline rules evaluated directly.")
        with col_d2:
            st.markdown("**Extracted Content Analyzed:**")
            st.text(extracted_text or "No content text available.")