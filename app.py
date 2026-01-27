import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="CYBERML Project - IoT Security Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ POUR LE LOOK "CYBER" ---
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .sidebar .sidebar-content { background: #262730; }
    h1 { color: #00ff41; font-family: 'Courier New', monospace; }
    h2, h3 { color: #ffffff; font-family: 'Helvetica', sans-serif; }
    .stMetric { background-color: #1f2937; border-radius: 10px; padding: 10px; border: 1px solid #374151; }
    .stAlert { background-color: #374151; color: white; }
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES (MOCKUP DATA) ---
# NOTE: Pour la démo, on simule des comportements réalistes basés sur votre notebook.
# Si vous avez exporté vos modèles (joblib), vous pouvez les charger ici.

@st.cache_data
def load_mock_data():
    """Génère un faux dataset représentatif de CIC IoT-DIAD pour la démo"""
    columns = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 
               'Total Length of Fwd Packet', 'Flow Bytes/s', 'Flow Packets/s', 
               'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean']
    
    # Génération de données bénignes
    benign = pd.DataFrame(np.random.normal(10, 2, (100, len(columns))), columns=columns)
    benign['Label'] = 'Benign'
    
    # Génération d'attaques (DDoS = Haut débit, BruteForce = Courtes durées repetées)
    ddos = pd.DataFrame(np.random.normal(50, 10, (50, len(columns))), columns=columns)
    ddos['Flow Duration'] = ddos['Flow Duration'] * 100
    ddos['Label'] = 'DDoS'
    
    mirai = pd.DataFrame(np.random.normal(30, 5, (50, len(columns))), columns=columns)
    mirai['Label'] = 'Mirai'
    
    return pd.concat([benign, ddos, mirai]).reset_index(drop=True)

def simulate_prediction(input_data, noise_level=0.0):
    """
    Simule la réponse de votre LightGBM/ResNet.
    Si noise_level (attaque adverse) augmente, la confiance diminue ou la classe change.
    """
    # Logique simplifiée pour la démo :
    # Si Flow Duration est haut -> Tendance Malveillante
    score_malicious = (input_data['Flow Duration'] / 100) * 0.7 + (input_data['Flow Bytes/s'] / 50) * 0.3
    
    # Application de l'attaque adverse (bruit qui réduit le score de détection)
    effective_score = score_malicious - (noise_level * 5.0) # L'attaque réduit le score malveillant
    
    prob_attack = 1 / (1 + np.exp(-effective_score)) # Sigmoid like
    
    if prob_attack > 0.5:
        return "ATTACK DETECTED", prob_attack, "red"
    else:
        return "BENIGN TRAFFIC", prob_attack, "green"

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/plasticine/100/000000/security-checked.png", width=100)
st.sidebar.title("IoT Sentinel 3000")
st.sidebar.info("Model: **LightGBM Ensemble**\n\nDataset: **CIC IoT-DIAD 2024**")

page = st.sidebar.radio("Navigation", ["Overview", "Live Detection", "Adversarial Lab (Bonus)"])

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("🛡️ Project Overview & Analytics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows Analyzed", "69,876", "+Stratified")
    col2.metric("Accuracy (Test)", "99.2%", "LightGBM")
    col3.metric("Attack Ratio", "85.69%", "High Threat")
    
    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Class Distribution (Stratified Sample)")
        # Données issues de votre rapport
        labels = ['Benign', 'Recon', 'Spoofing', 'Mirai', 'DDoS', 'DoS', 'Web', 'BruteForce']
        values = [10000, 10000, 9999, 9976, 9943, 8325, 8014, 3619]
        fig = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Model Performance")
        perf_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'ResNet-MLP'],
            'MCC Score': [0.6016, 0.6170, 0.6208, 0.4888]
        })
        fig_bar = px.bar(perf_data, x='Model', y='MCC Score', color='MCC Score', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE 2: LIVE DETECTION ---
elif page == "Live Detection":
    st.title("📡 Live Traffic Analysis")
    st.markdown("Simulating real-time packet flow ingestion from IoT Gateway.")
    
    if st.button("Start Capture Simulation"):
        # Placeholder pour les logs
        log_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        data = load_mock_data()
        history = []
        
        progress_bar = st.progress(0)
        
        for i in range(20):
            # Simulation d'arrivée de paquets
            sample = data.sample(1).iloc[0]
            pred, prob, color = simulate_prediction(sample)
            
            history.append({'Time': i, 'Threat Level': prob})
            
            # Affichage dynamique
            with log_placeholder.container():
                c1, c2, c3 = st.columns([1, 2, 1])
                c1.write(f"**Packet ID:** #{1024+i}")
                c2.write(f"**Flow Duration:** {sample['Flow Duration']:.2f} ms")
                if color == "red":
                    c3.error(f"🚨 {pred} ({prob:.2%})")
                else:
                    c3.success(f"✅ {pred} ({prob:.2%})")
            
            # Graphique temps réel
            df_hist = pd.DataFrame(history)
            fig = px.area(df_hist, x='Time', y='Threat Level', range_y=[0, 1])
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            time.sleep(0.1) # Vitesse de simulation
            progress_bar.progress((i+1)/20)

# --- PAGE 3: ADVERSARIAL LAB (LE BONUS) ---
elif page == "Adversarial Lab (Bonus)":
    st.title("🧪 Adversarial Attack Simulator")
    st.markdown("""
    **Objective:** Demonstrate robustness against FGSM/Noise attacks.
    Select a malicious sample and try to fool the classifier by adding perturbations.
    """)
    
    col_ctrl, col_viz = st.columns([1, 2])
    
    data = load_mock_data()
    # On prend seulement les attaques pour la démo
    attack_samples = data[data['Label'] != 'Benign']
    
    with col_ctrl:
        st.subheader("1. Select Threat")
        sample_id = st.selectbox("Choose a captured packet:", attack_samples.index[:10])
        original_sample = attack_samples.loc[sample_id]
        
        st.subheader("2. Attack Configuration")
        attack_type = st.selectbox("Attack Method:", ["Random Noise", "FGSM (White-box)", "PGD"])
        epsilon = st.slider("Epsilon (Perturbation Strength)", 0.0, 1.0, 0.0, 0.05)
        
        st.info(f"Targeting Model parameters with **{attack_type}**...")
        
    with col_viz:
        st.subheader("3. Real-time Impact Analysis")
        
        # Prédiction sans bruit
        orig_res, orig_prob, orig_color = simulate_prediction(original_sample, noise_level=0.0)
        
        # Prédiction AVEC bruit (Attaque)
        # Note: Dans la vraie vie, l'epsilon FGSM a un impact plus fort que le random noise
        impact_factor = 2.0 if attack_type == "FGSM (White-box)" else 0.8
        adv_res, adv_prob, adv_color = simulate_prediction(original_sample, noise_level=epsilon * impact_factor)
        
        # Affichage cote à cote
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Original Prediction")
            fig_gauge_orig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = orig_prob * 100,
                title = {'text': "Malicious Probability"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "red" if orig_prob > 0.5 else "green"}}
            ))
            fig_gauge_orig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_orig, use_container_width=True)
            if orig_res == "ATTACK DETECTED":
                st.error("Correctly Identified as Attack")
            else:
                st.warning("False Negative")

        with c2:
            st.markdown(f"#### Under {attack_type}")
            fig_gauge_adv = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = adv_prob * 100,
                title = {'text': "Malicious Probability"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "red" if adv_prob > 0.5 else "green"}}
            ))
            fig_gauge_adv.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_adv, use_container_width=True)
            
            if adv_res == "BENIGN TRAFFIC":
                st.success("🚨 MODEL EVADED! (Attack successful)")
            else:
                st.error("Model Resisted")

    # Visualisation des Features perturbées
    st.markdown("---")
    st.subheader("Feature Space Perturbation")
    
    # Création d'un graph pour montrer la différence entre Original et Adversarial
    # On simule la perturbation sur les données pour le visuel
    perturbed_sample = original_sample.copy()
    if isinstance(perturbed_sample['Flow Duration'], (int, float)):
         perturbed_sample['Flow Duration'] = perturbed_sample['Flow Duration'] * (1.0 - epsilon)
         perturbed_sample['Flow Bytes/s'] = perturbed_sample['Flow Bytes/s'] * (1.0 + epsilon)
    
    # Normalisation pour affichage radar
    categories = ['Flow Duration', 'Flow Bytes/s', 'Packet Length Mean', 'Flow IAT Mean']
    # Valeurs fictives normalisées pour le radar chart
    values_orig = [0.8, 0.7, 0.6, 0.9] 
    values_adv = [0.8 - epsilon, 0.7 + epsilon, 0.6 + epsilon/2, 0.9 - epsilon]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values_orig, theta=categories, fill='toself', name='Original Flow'))
    fig_radar.add_trace(go.Scatterpolar(r=values_adv, theta=categories, fill='toself', name='Adversarial Flow'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    
    st.plotly_chart(fig_radar, use_container_width=True)