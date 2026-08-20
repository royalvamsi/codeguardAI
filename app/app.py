"""
CodeGuard AI - Python Code Error Classification & Bug Detection System
Professional AI Developer Tool & Diagnostic IDE Interface.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src.predictor import predict_code
from src.suggestion_engine import ERROR_KNOWLEDGE_BASE

# ============================================================
# Page Configuration & Global Theme
# ============================================================

st.set_page_config(
    page_title="CodeGuard AI — Python Diagnostic Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Developer Tool CSS (Dark Modern IDE Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, textarea {
        font-family: 'JetBrains Mono', Consolas, monospace !important;
    }

    /* Top App Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0 1.2rem 0;
        border-bottom: 1px solid #21262D;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #F0F6FC;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: #8B949E;
        font-weight: 500;
    }
    .status-pill-ready {
        background: rgba(35, 134, 54, 0.15);
        color: #3FB950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Code Window Frame */
    .code-frame {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 0.8rem;
    }
    .code-frame-header {
        background: #161B22;
        padding: 8px 14px;
        border-bottom: 1px solid #30363D;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        color: #8B949E;
        font-weight: 600;
    }
    .window-dots {
        display: flex;
        gap: 6px;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-red { background: #FF5F56; }
    .dot-yellow { background: #FFBD2E; }
    .dot-green { background: #27C93F; }

    /* Diagnostic Result Cards */
    .res-card-empty {
        background: #0D1117;
        border: 1px dashed #30363D;
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
        color: #8B949E;
    }
    .res-card-error {
        background: #0D1117;
        border: 1px solid rgba(248, 81, 73, 0.4);
        border-left: 4px solid #F85149;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .res-card-clean {
        background: #0D1117;
        border: 1px solid rgba(63, 185, 80, 0.4);
        border-left: 4px solid #3FB950;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    
    .res-title-error {
        color: #FF7B72;
        font-size: 1.3rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.4rem;
    }
    .res-title-clean {
        color: #56D364;
        font-size: 1.3rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.4rem;
    }

    /* Section Cards */
    .section-box {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .section-tag {
        font-size: 0.72rem;
        font-weight: 700;
        color: #58A6FF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.35rem;
    }
    .section-content {
        font-size: 0.92rem;
        color: #C9D1D9;
        line-height: 1.5;
    }

    /* Badges */
    .badge-static {
        background: rgba(56, 139, 253, 0.15);
        color: #58A6FF;
        border: 1px solid rgba(56, 139, 253, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ml {
        background: rgba(187, 128, 255, 0.15);
        color: #D2A8FF;
        border: 1px solid rgba(187, 128, 255, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Bottom Architecture Ribbon */
    .arch-ribbon {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px 16px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-size: 0.78rem;
        font-weight: 600;
        color: #8B949E;
        margin-top: 2rem;
    }
    .arch-step {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #C9D1D9;
    }
    .arch-arrow {
        color: #484F58;
    }

    /* Metric Cards */
    .kpi-tile {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-num {
        font-size: 2rem;
        font-weight: 800;
        color: #F0F6FC;
    }
    .kpi-name {
        font-size: 0.75rem;
        font-weight: 700;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }

    /* Fix Button Text Wrapping on Example Chips */
    div[data-testid="stHorizontalBlock"] button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 4px 8px !important;
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Quick Preset Library (Categorized by Error Type)
# ============================================================

SNIPPET_PRESETS = {
    "NameError": """# Variable referenced before definition
def calculate_subtotal(unit_price, count):
    total = unit_price * count + shipping_charge
    return total

print("Result:", calculate_subtotal(25.0, 4))""",

    "TypeError": """# Incompatible operand types (str + int)
name = "Royal"
age = 20

result = name + age
print(result)""",

    "ZeroDivisionError": """# Division by zero
a = 100
b = 0

result = a / b
print(result)""",

    "ValueError": """# Invalid numeric conversion
value = "hello"

number = int(value)
print(number)""",

    "IndexError": """# Accessing list element outside valid range
matrix_row = [101, 102, 103, 104]
element = matrix_row[8]
print("Found element:", element)""",

    "KeyError": """# Accessing non-existent dictionary key
user_profile = {
    "username": "coder_99",
    "role": "developer"
}
print("Account status:", user_profile["subscription_tier"])""",

    "AttributeError": """# Invalid method call on built-in object
log_message = "System initialized successfully"
log_message.append(" [OK]")
print(log_message)""",

    "SyntaxError": """# Missing colon in function declaration
def process_data(records)
    for r in records:
        print(r)

process_data([1, 2, 3])""",

    "Clean Code": """# Clean, validated Python implementation
def compute_factorial(n):
    if n <= 1:
        return 1
    return n * compute_factorial(n - 1)

print("5! =", compute_factorial(5))""",
}


# ============================================================
# Sidebar Navigation
# ============================================================

with st.sidebar:
    st.markdown("### 🛡️ CodeGuard AI")
    st.caption("AI-Powered Code Error Classifier & Diagnostic Assistant")
    
    st.markdown("---")
    
    app_tab = st.radio(
        "Navigation Menu",
        ["🔍 Code Analyzer", "📊 Model Benchmarks", "📖 Error Knowledge Base"],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("##### ⚙️ System Status")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #8B949E; line-height: 1.8;'>
        <div>● <span style='color:#3FB950;'>Engine Active</span></div>
        <div>● 10 Target Classes</div>
        <div>● Zero Train/Test Leakage</div>
        <div>● PyMETA Corpus (35.5k clean)</div>
        <div>● Model: Hybrid TF-IDF + SVM</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("NIELIT AI/ML Major Project • 2026")


# ============================================================
# Tab 1: Analyzer (Main Workspace)
# ============================================================

if app_tab == "🔍 Code Analyzer":
    
    # Top Status Bar
    st.markdown("""
    <div class='top-bar'>
        <div>
            <div class='brand-title'>🛡️ CodeGuard AI <span style='font-size: 0.85rem; font-weight: 500; color: #8B949E;'>/ Diagnostic Workspace</span></div>
            <div class='brand-subtitle'>Static AST rule validation + Hybrid machine-learning error classification for Python</div>
        </div>
        <div>
            <span class='status-pill-ready'>● Engine Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State for Editor
    if "editor_code" not in st.session_state:
        st.session_state["editor_code"] = ""
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "execution_time" not in st.session_state:
        st.session_state["execution_time"] = None

    # Workspace Columns
    col_editor, col_diagnostic = st.columns([1.1, 0.9], gap="large")

    with col_editor:
        # Code Window Top Bar
        st.markdown("""
        <div class='code-frame'>
            <div class='code-frame-header'>
                <div class='window-dots'>
                    <span class='dot dot-red'></span>
                    <span class='dot dot-yellow'></span>
                    <span class='dot dot-green'></span>
                </div>
                <div><span>main.py</span></div>
                <div><span>Python 3.x</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        code_text = st.text_area(
            "Source Code",
            value=st.session_state["editor_code"],
            height=300,
            placeholder="# Paste or type Python code here...\n\ndef calculate(a, b):\n    return a + b\n\nprint(calculate(10, 20))",
            label_visibility="collapsed",
            key="code_editor_area"
        )
        st.session_state["editor_code"] = code_text

        # Action Buttons Row
        c_btn1, c_btn2, c_spacer = st.columns([1.2, 0.9, 1.4])
        with c_btn1:
            analyze_clicked = st.button("⚡ Analyze Code", type="primary", use_container_width=True)
        with c_btn2:
            clear_clicked = st.button("🔄 Clear", use_container_width=True)
            if clear_clicked:
                st.session_state["editor_code"] = ""
                st.session_state["analysis_result"] = None
                st.rerun()

        # Quick Example Chips (2 Rows for clean, non-wrapped buttons)
        st.write("")
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #8B949E; margin-bottom: 6px; letter-spacing:0.5px;'>TRY AN EXAMPLE:</div>", unsafe_allow_html=True)
        
        row1_keys = ["NameError", "TypeError", "ZeroDivisionError", "ValueError"]
        row2_keys = ["IndexError", "KeyError", "AttributeError", "SyntaxError", "Clean Code"]

        r1_cols = st.columns(len(row1_keys))
        for idx, key_name in enumerate(row1_keys):
            with r1_cols[idx]:
                if st.button(f"{key_name}", key=f"chip_{key_name}", use_container_width=True):
                    st.session_state["editor_code"] = SNIPPET_PRESETS[key_name]
                    st.session_state["analysis_result"] = None
                    st.rerun()

        r2_cols = st.columns(len(row2_keys))
        for idx, key_name in enumerate(row2_keys):
            with r2_cols[idx]:
                if st.button(f"{key_name}", key=f"chip_{key_name}", use_container_width=True):
                    st.session_state["editor_code"] = SNIPPET_PRESETS[key_name]
                    st.session_state["analysis_result"] = None
                    st.rerun()

    # Trigger Analysis Logic
    if analyze_clicked:
        if not code_text.strip():
            st.warning("⚠️ Source code is empty. Please enter or paste Python code to analyze.")
            st.session_state["analysis_result"] = None
        else:
            t0 = time.time()
            res = predict_code(code_text)
            elapsed_ms = (time.time() - t0) * 1000
            st.session_state["analysis_result"] = res
            st.session_state["execution_time"] = elapsed_ms

    # Diagnostic Output Panel
    with col_diagnostic:
        res = st.session_state.get("analysis_result")
        exec_ms = st.session_state.get("execution_time")

        if res is None:
            st.markdown("""
            <div class='res-card-empty'>
                <div style='font-size: 1.8rem; margin-bottom: 0.6rem;'>🛡️</div>
                <div style='font-size: 1.05rem; font-weight: 700; color: #C9D1D9; margin-bottom: 0.4rem;'>Ready to Inspect</div>
                <div style='font-size: 0.85rem; line-height: 1.5;'>Paste Python code in the editor and click <strong>Analyze Code</strong>.<br>CodeGuard AI will perform AST syntax verification, symbol resolution, bounds checking, and hybrid ML classification.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            error_type = res["error_type"]
            source = res["source"]
            is_clean = error_type in (None, "No error")
            exp = res["explanation"]

            # Result Header Card
            if is_clean:
                st.markdown(f"""
                <div class='res-card-clean'>
                    <div class='res-title-clean'>🟢 No High-Confidence Error Detected</div>
                    <div style='color: #8B949E; font-size: 0.85rem;'>Analysis verified in {exec_ms:.1f} ms • No syntax violations or common runtime error patterns identified</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='res-card-error'>
                    <div class='res-title-error'>● Error Detected: {error_type}</div>
                    <div style='color: #8B949E; font-size: 0.85rem;'>{exp.get('category', 'Runtime Exception')} • Analysis completed in {exec_ms:.1f} ms</div>
                </div>
                """, unsafe_allow_html=True)

            # Metadata Row
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                badge_html = f"<span class='badge-static'>{res['method_label']}</span>" if source == "static_analysis" else f"<span class='badge-ml'>{res['method_label']}</span>"
                st.markdown(f"<div style='font-size: 0.75rem; color: #8B949E; text-transform: uppercase; font-weight: 700; margin-bottom: 3px;'>Detection Layer</div>{badge_html}", unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"<div style='font-size: 0.75rem; color: #8B949E; text-transform: uppercase; font-weight: 700; margin-bottom: 3px;'>Confidence Indicator</div><code style='font-size: 0.85rem;'>{res['confidence_label']}</code>", unsafe_allow_html=True)

            st.write("")

            # Diagnosis Section Box
            st.markdown(f"""
            <div class='section-box'>
                <div class='section-tag'>Why This Happens</div>
                <div class='section-content'>{res['message']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation Section Box
            st.markdown(f"""
            <div class='section-box'>
                <div class='section-tag'>Recommended Action</div>
                <div class='section-content'>{res['suggestion']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Decision Distribution (for ML)
            if source == "machine_learning" and res.get("decision_scores"):
                with st.expander("📊 View Model Decision Function Breakdown"):
                    top_5 = list(res["decision_scores"].items())[:5]
                    df_scores = pd.DataFrame(top_5, columns=["Error Class", "Score"])
                    st.bar_chart(df_scores.set_index("Error Class"), height=140)
                    st.caption("Softmax-normalized relative decision scores from the Linear SVM hyperplanes.")

            # Buggy vs Fixed Pattern Comparison
            if not is_clean and exp.get("buggy_example"):
                with st.expander("💡 View Corrected Pattern Example"):
                    e_col1, e_col2 = st.columns(2)
                    with e_col1:
                        st.markdown("<span style='font-size: 0.75rem; font-weight:700; color:#FF7B72;'>BUGGY PATTERN:</span>", unsafe_allow_html=True)
                        st.code(exp["buggy_example"], language="python")
                    with e_col2:
                        st.markdown("<span style='font-size: 0.75rem; font-weight:700; color:#56D364;'>FIXED PATTERN:</span>", unsafe_allow_html=True)
                        st.code(exp["fixed_example"], language="python")

    # Bottom Architectural Pipeline Ribbon
    st.markdown("""
    <div class='arch-ribbon'>
        <div class='arch-step'><span>⚡</span> AST Static Rules</div>
        <div class='arch-arrow'>→</div>
        <div class='arch-step'><span>🔤</span> Hybrid TF-IDF (Word + Char)</div>
        <div class='arch-arrow'>→</div>
        <div class='arch-step'><span>🤖</span> Linear SVM Classifier</div>
        <div class='arch-arrow'>→</div>
        <div class='arch-step'><span>💡</span> Explainable Suggestions</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Tab 2: Model Performance & Empirical Benchmarks
# ============================================================

elif app_tab == "📊 Model Benchmarks":
    st.markdown("""
    <div class='top-bar'>
        <div>
            <div class='brand-title'>◈ Model Performance & Empirical Benchmarks</div>
            <div class='brand-subtitle'>Rigorous multi-model evaluation on 7,117 held-out PyMETA test samples (zero train/test overlap)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("<div class='kpi-tile'><div class='kpi-num'>83.84%</div><div class='kpi-name'>Test Accuracy</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='kpi-tile'><div class='kpi-num'>0.5750</div><div class='kpi-name'>Macro F1 Score</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='kpi-tile'><div class='kpi-num'>0.8443</div><div class='kpi-name'>Weighted F1 Score</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div class='kpi-tile'><div class='kpi-num'>7,117</div><div class='kpi-name'>Held-Out Test Set</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    # Experimental Comparison Table
    st.markdown("#### 🏆 Experimental Model Benchmark Table")
    comp_csv_path = PROJECT_ROOT / "results" / "advanced_model_comparison.csv"
    if comp_csv_path.exists():
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.2%}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Weighted F1": "{:.4f}",
                "Training Time (s)": "{:.2f}s",
                "Inference Time (s)": "{:.2f}s",
            }),
            use_container_width=True
        )

    st.write("")
    st.markdown("---")

    # Confusion Matrix & Class Breakdown
    col_cm, col_class = st.columns([1.1, 0.9], gap="large")

    with col_cm:
        st.markdown("#### 🎯 Normalized Confusion Matrix")
        cm_path = PROJECT_ROOT / "results" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Normalized Confusion Matrix for Hybrid TF-IDF + Linear SVM", use_container_width=True)

    with col_class:
        st.markdown("#### 📈 Class-wise Performance Breakdown")
        st.markdown("""
        | Error Class | Precision | Recall | F1-Score | Support |
        | :--- | :---: | :---: | :---: | :---: |
        | **No error** | 0.93 | 0.89 | **0.91** | 5,848 |
        | **KeyError** | 0.48 | 0.88 | **0.62** | 72 |
        | **EOFError** | 0.56 | 0.62 | **0.59** | 53 |
        | **RecursionError** | 0.45 | 0.85 | **0.59** | 47 |
        | **IndexError** | 0.48 | 0.70 | **0.57** | 74 |
        | **NameError** | 0.54 | 0.52 | **0.53** | 485 |
        | **TypeError** | 0.48 | 0.52 | **0.50** | 387 |
        | **ValueError** | 0.45 | 0.56 | **0.50** | 36 |
        | **UnboundLocalError** | 0.40 | 0.65 | **0.49** | 93 |
        | **AttributeError** | 0.38 | 0.55 | **0.44** | 22 |
        """)

        st.markdown("""
        <div style='background:#161B22; border:1px solid #30363D; border-radius:6px; padding:12px; font-size:0.82rem; color:#8B949E; margin-top:12px;'>
            <strong style='color:#C9D1D9;'>Academic Notes:</strong><br>
            • <strong>Feature Representation:</strong> FeatureUnion combining word n-grams (1, 2) and character n-grams (2, 5).<br>
            • <strong>Deduplication:</strong> 13,007 cross-split duplicate submissions removed to ensure strict leakage control.<br>
            • <strong>Decision Metric:</strong> Selected by Macro F1 to balance minority error classes against the dominant clean code baseline.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# Tab 3: Error Diagnostic Knowledge Base
# ============================================================

elif app_tab == "📖 Error Knowledge Base":
    st.markdown("""
    <div class='top-bar'>
        <div>
            <div class='brand-title'>▣ Error Diagnostic Knowledge Base</div>
            <div class='brand-subtitle'>Structured reference catalog for Python syntax violations and runtime error classes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for err_name, info in ERROR_KNOWLEDGE_BASE.items():
        with st.expander(f"📌 {info['title']}", expanded=(err_name in ["SyntaxError", "NameError"])):
            st.markdown(f"**Category**: `{info['category']}`")
            st.markdown(f"**Technical Description**: {info['description']}")
            
            st.markdown("**Common Root Causes:**")
            for cause in info["common_causes"]:
                st.markdown(f"- {cause}")
                
            st.markdown("**Corrective Recommendations:**")
            for sugg in info["suggestions"]:
                st.markdown(f"- {sugg}")

            if info.get("buggy_example"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<span style='font-size: 0.75rem; font-weight:700; color:#FF7B72;'>BUGGY CODE:</span>", unsafe_allow_html=True)
                    st.code(info["buggy_example"], language="python")
                with c2:
                    st.markdown("<span style='font-size: 0.75rem; font-weight:700; color:#56D364;'>CORRECTED CODE:</span>", unsafe_allow_html=True)
                    st.code(info["fixed_example"], language="python")
