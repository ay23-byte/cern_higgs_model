import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import roc_curve, roc_auc_score

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="CERN Higgs Dashboard", layout="wide", page_icon="⚛️")

# Custom CSS for a professional, scientific look
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .big-font {font-size:24px !important; font-weight: bold; color: #0e1117;}
</style>
""", unsafe_allow_html=True)

st.title("⚛️ CERN Higgs Boson Identification")
st.markdown("Interactive Machine Learning Dashboard for Particle Physics Analysis")

# --- BEGINNER'S GUIDE EXPANDER ---
with st.expander("👶 Beginner's Guide: What am I looking at? (Click to expand)"):
    st.markdown("""
    **The Problem:**
    Imagine trying to find a specific type of red apple in a massive pile of red and green apples, but the red ones are extremely rare and look almost identical to the green ones. That is what physicists do at CERN!
    *   🔴 **Signal (Higgs Boson):** A rare, heavy particle we are looking for.
    *   🔵 **Background (Z Boson):** A common, lighter particle that creates "noise."
    
    **The Solution:**
    We trained an Artificial Intelligence (Machine Learning) to look at the "debris" of particle collisions and guess which ones are Higgs bosons. 
    
    **How to use this dashboard:**
    1. Look at the **Physics Distributions** to see how much the two particles overlap.
    2. Watch the **2D Decision Space** to see how the AI draws a boundary to separate them.
    3. Use the **Threshold Slider** on the left to change how "strict" the AI is!
    """)

st.markdown("---")

# ==========================================
# 2. INSTANT MODEL LOADING
# ==========================================
@st.cache_resource
def load_pretrained_model():
    model = joblib.load('outputs/models/higgs_rf_model.pkl')
    feature_names = joblib.load('outputs/models/feature_names.pkl')
    return model, feature_names

with st.spinner("Loading Physics Model..."):
    model, feature_names = load_pretrained_model()

# ==========================================
# 3. LOAD DATA FOR VISUALIZATION
# ==========================================
@st.cache_data
def load_dashboard_data():
    df = pd.read_csv('data/raw/training.csv', nrows=15000)
    df.replace(-999.0, np.nan, inplace=True)
    y = df['Label'].apply(lambda x: 1 if x == 's' else 0)
    X = df.drop(['EventId', 'Label', 'Weight'], axis=1)
    X.fillna(X.median(), inplace=True)
    return X, y

X_dash, y_dash = load_dashboard_data()

# ==========================================
# 4. SIDEBAR CONTROLS & METRICS
# ==========================================
st.sidebar.header("⚙️ Analysis Controls")

# The Security Scanner Analogy
st.sidebar.markdown("""
**🛡️ The Security Scanner Analogy:**
Think of the slider like an airport security scanner. 
*   **Slide Left (Low Threshold):** The scanner beeps at *everything*. You catch all the bad guys (Higgs), but you also stop a lot of innocent people (Z Bosons). 
*   **Slide Right (High Threshold):** The scanner only beeps if it's 100% sure. You stop almost no innocent people, but some bad guys might slip through!
""")

threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.50, 0.01)

# Calculate predictions based on slider (MUST happen before metrics)
y_scores = model.predict_proba(X_dash)[:, 1]
y_pred = (y_scores >= threshold).astype(int)

# Calculate Confusion Matrix values
tp = np.sum((y_pred == 1) & (y_dash == 1))
fp = np.sum((y_pred == 1) & (y_dash == 0))
fn = np.sum((y_pred == 0) & (y_dash == 1))
tn = np.sum((y_pred == 0) & (y_dash == 0))

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Current Performance")

# Beginner-friendly metrics with tooltips
st.sidebar.metric(
    label="Higgs Caught (Signal Efficiency)", 
    value=f"{tp / (tp + fn):.1%}" if (tp + fn) > 0 else "0%",
    help="Out of 100 real Higgs bosons, how many did the AI correctly find?"
)
st.sidebar.metric(
    label="Noise Blocked (Background Rejection)", 
    value=f"{tn / (tn + fp):.1%}" if (tn + fp) > 0 else "0%",
    help="Out of 100 Z bosons (noise), how many did the AI correctly ignore?"
)
st.sidebar.metric(
    label="Accuracy of 'Higgs' Claims (Precision)", 
    value=f"{tp / (tp + fp):.1%}" if (tp + fp) > 0 else "0%",
    help="When the AI says 'This is a Higgs', how often is it actually right?"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Animations")
play_animation = st.sidebar.button("▶️ Play Threshold Sweep Animation")

# ==========================================
# 5. MAIN VISUALIZATIONS (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Physics Distributions", "🌌 2D Decision Space", "📈 ROC Curve"])

# --- TAB 1: PHYSICS DISTRIBUTIONS ---
with tab1:
    st.subheader("The Overlap Problem")
    st.markdown("""
    This chart shows the 'fingerprints' of the two particles based on their energy (Transverse Mass).
    *   🔵 **Blue:** The common Z Boson (Noise).
    *   🔴 **Red:** The rare Higgs Boson (Signal).
    
    *Notice how they overlap in the middle? Because they look so similar, a human cannot easily separate them. This is why we need AI!*
    """)
    
    feature_to_plot = 'DER_mass_transverse_met_lep'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(X_dash[y_dash == 0][feature_to_plot], bins=50, alpha=0.5, color='blue', label='True Background (Z Boson)', density=True)
    ax.hist(X_dash[y_dash == 1][feature_to_plot], bins=50, alpha=0.5, color='red', label='True Signal (Higgs Boson)', density=True)
    
    ax.set_xlabel('Transverse Mass (GeV)', fontsize=12)
    ax.set_ylabel('Normalized Frequency', fontsize=12)
    ax.set_title('Overlap of Higgs Signal and Z Boson Background', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# --- TAB 2: 2D DECISION SPACE (WITH ANIMATION) ---
with tab2:
    st.subheader("How the AI Separates Them")
    st.markdown("""
    This is a map of the particle collisions using the two most important physical features. 
    *   🔵 **Blue dots:** Events the AI thinks are Background (Z Boson).
    *   🔴 **Red dots:** Events the AI thinks are Signal (Higgs).
    
    **Try the animation button on the left!** Watch how the AI draws an invisible boundary to separate the red dots from the blue cloud.
    """)
    
    x_feat = 'DER_mass_transverse_met_lep'
    y_feat = 'DER_mass_MMC'
    
    if play_animation:
        st.info("Running animation... (Sweeping threshold from 0.0 to 1.0)")
        plot_placeholder = st.empty()
        
        anim_X = X_dash.head(3000)
        anim_y_scores = y_scores[:3000]
        
        for t in np.arange(0.01, 1.0, 0.03):
            anim_y_pred = (anim_y_scores >= t).astype(int)
            
            fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
            ax_anim.scatter(anim_X[anim_y_pred == 0][x_feat], anim_X[anim_y_pred == 0][y_feat], 
                            c='blue', alpha=0.4, s=15, label='Predicted Background')
            ax_anim.scatter(anim_X[anim_y_pred == 1][x_feat], anim_X[anim_y_pred == 1][y_feat], 
                            c='red', alpha=0.7, s=15, label='Predicted Signal (Higgs)')
            
            ax_anim.set_xlabel(x_feat, fontsize=12)
            ax_anim.set_ylabel(y_feat, fontsize=12)
            ax_anim.set_title(f'Model Decision Boundary (Threshold = {t:.2f})', fontsize=14)
            ax_anim.set_xlim(-50, 450)
            ax_anim.set_ylim(-50, 850)
            ax_anim.legend()
            ax_anim.grid(True, alpha=0.3)
            
            plot_placeholder.pyplot(fig_anim)
            plt.close(fig_anim)
            time.sleep(0.05) # Fast, smooth animation
            
        st.success("Animation complete! Use the slider to explore manually.")

    else:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.scatter(X_dash[y_pred == 0][x_feat], X_dash[y_pred == 0][y_feat], 
                    c='blue', alpha=0.3, s=15, label='Predicted Background')
        ax2.scatter(X_dash[y_pred == 1][x_feat], X_dash[y_pred == 1][y_feat], 
                    c='red', alpha=0.6, s=15, label='Predicted Signal (Higgs)')
        
        ax2.set_xlabel(x_feat, fontsize=12)
        ax2.set_ylabel(y_feat, fontsize=12)
        ax2.set_title(f'Model Decision Boundary (Threshold = {threshold})', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# --- TAB 3: ROC CURVE ---
with tab3:
    st.subheader("The Trade-Off Curve (ROC)")
    st.markdown("""
    This curve shows the AI's overall skill at separating the particles. 
    *   The **Orange Line** is our AI. The closer it hugs the top-left corner, the smarter it is.
    *   The **Dashed Blue Line** is random guessing (like flipping a coin).
    *   The **Red Dot** shows exactly where your slider is currently set.
    
    *Our AI scored an AUC of ~0.90, meaning it is highly effective at telling these particles apart!*
    """)
    
    fpr, tpr, thresholds = roc_curve(y_dash, y_scores)
    auc = roc_auc_score(y_dash, y_scores)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    
    idx = np.argmin(np.abs(thresholds - threshold))
    ax3.plot(fpr[idx], tpr[idx], 'ro', markersize=10, label=f'Current Threshold ({threshold})')
    
    ax3.set_xlabel('False Positive Rate (Noise let through)')
    ax3.set_ylabel('True Positive Rate (Higgs caught)')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

# ==========================================
# 6. FOOTER
# ==========================================
st.markdown("---")
st.caption("Built for CERN Summer Student Programme Application | Model: Random Forest (200 estimators) | Data: CERN Open Data Portal")