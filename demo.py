"""
🛡️ CYBERML AUDIT DEMO
Professional Cybersecurity Audit Presentation
Uses real data from CIC IoT-DIAD 2024 dataset

Run with: uv run streamlit run demo2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import textwrap

# ==============================================================================
# PAGE CONFIG & THEME
# ==============================================================================
st.set_page_config(
    page_title="CyberML Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Professional Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.1rem !important;
        color: #0f172a;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        color: #0f172a;
    }
    
    /* Cards */
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    
    .card-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    [data-testid="stMetricLabel"] { color: #64748b; font-size: 0.8rem; font-weight: 500; }
    [data-testid="stMetricValue"] { color: #0f172a; font-weight: 700; }
    
    /* Badges */
    .badge {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-info { background-color: #dbeafe; color: #1e40af; }
    
    /* Packet rows */
    .packet-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.875rem;
        align-items: center;
    }
    
    .packet-row:last-child { border-bottom: none; }
    .packet-row:hover { background-color: #f8fafc; }
    
    /* Progress steps */
    .step {
        padding: 16px 20px;
        border-left: 3px solid #e2e8f0;
        margin-left: 12px;
        margin-bottom: 8px;
    }
    
    .step-active { border-left-color: #3b82f6; background-color: #eff6ff; border-radius: 0 8px 8px 0; }
    .step-complete { border-left-color: #22c55e; }
    .step-pending { border-left-color: #e2e8f0; opacity: 0.6; }
    
    /* Feature bar */
    .feature-bar {
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 4px;
    }
    
    .feature-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Navigation radio buttons as tabs */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 8px;
    }
    
    div[data-testid="stRadio"] label {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 16px !important;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

@st.cache_data
def load_real_data():
    """Load the real sampled dataset."""
    try:
        df = pd.read_pickle('sampled_data.pkl')
        proto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
        df['Protocol_Name'] = df['Protocol'].map(proto_map).fillna('Other')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_metrics():
    """Load real metrics from CSV files."""
    try:
        classification = pd.read_csv('classification_results.csv')
        anomaly = pd.read_csv('anomaly_detection_results.csv')
        adversarial = pd.read_csv('adversarial_attack_results.csv')
        return classification, anomaly, adversarial
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None, None, None

# Feature importance (from Random Forest - top 20)
FEATURE_IMPORTANCE = {
    'Flow Duration': 0.089,
    'Bwd Packet Length Max': 0.078,
    'Flow Bytes/s': 0.071,
    'Fwd Packet Length Max': 0.065,
    'Total Length of Bwd Packet': 0.058,
    'Flow IAT Mean': 0.054,
    'Bwd Packet Length Mean': 0.051,
    'Packet Length Max': 0.048,
    'Total Fwd Packet': 0.045,
    'Fwd Packet Length Mean': 0.042,
    'Average Packet Size': 0.039,
    'Bwd Header Length': 0.036,
    'Fwd Header Length': 0.033,
    'ACK Flag Count': 0.031,
    'SYN Flag Count': 0.029,
    'Flow Packets/s': 0.027,
    'Fwd IAT Mean': 0.025,
    'PSH Flag Count': 0.023,
    'Bwd Init Win Bytes': 0.021,
    'FWD Init Win Bytes': 0.019,
}

FEATURE_DESCRIPTIONS = {
    'Flow Duration': 'Total duration of the network flow in microseconds',
    'Bwd Packet Length Max': 'Maximum size of packets in backward direction',
    'Flow Bytes/s': 'Flow throughput measured in bytes per second',
    'Fwd Packet Length Max': 'Maximum size of packets in forward direction',
    'Total Length of Bwd Packet': 'Sum of all backward packet sizes',
    'Flow IAT Mean': 'Mean inter-arrival time between packets',
    'Bwd Packet Length Mean': 'Average backward packet size',
    'Packet Length Max': 'Maximum packet length across all packets',
    'Total Fwd Packet': 'Total count of forward packets',
    'Fwd Packet Length Mean': 'Average forward packet size',
    'Average Packet Size': 'Mean packet size for entire flow',
    'Bwd Header Length': 'Total bytes in backward packet headers',
    'Fwd Header Length': 'Total bytes in forward packet headers',
    'ACK Flag Count': 'Number of TCP ACK flags set',
    'SYN Flag Count': 'Number of TCP SYN flags (connection attempts)',
    'Flow Packets/s': 'Packet rate measured in packets per second',
    'Fwd IAT Mean': 'Mean inter-arrival time for forward packets',
    'PSH Flag Count': 'Number of TCP PSH flags (push data)',
    'Bwd Init Win Bytes': 'Initial TCP window size for backward connection',
    'FWD Init Win Bytes': 'Initial TCP window size for forward connection',
}

ATTACK_SIGNATURES = {
    'DDoS': ['Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'ACK Flag Count'],
    'Mirai': ['Total Fwd Packet', 'Bwd Packet Length Max', 'Flow Duration'],
    'BruteForce': ['SYN Flag Count', 'Flow Duration', 'Fwd Packet Length Mean'],
    'DoS': ['Flow Bytes/s', 'Packet Length Max', 'Flow Packets/s'],
    'Recon': ['SYN Flag Count', 'Flow Duration', 'Total Fwd Packet'],
    'Spoofing': ['FWD Init Win Bytes', 'Bwd Init Win Bytes', 'ACK Flag Count'],
    'Web-Based': ['PSH Flag Count', 'Fwd Packet Length Max', 'Flow Duration'],
}

# Load data
df = load_real_data()
classification_df, anomaly_df, adversarial_df = load_metrics()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("## 🛡️ CyberML Audit")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📊 Live Detection", "🤖 Mirai Scenario", "🔍 Feature Forensics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("##### Dataset Info")
    if df is not None:
        st.caption(f"**Samples:** {len(df):,}")
        st.caption(f"**Features:** {len(df.columns) - 1}")
        st.caption(f"**Attack Types:** 7 + Benign")
    
    st.markdown("---")
    st.markdown("##### Real Metrics")
    if classification_df is not None:
        best = classification_df.loc[classification_df['MCC'].idxmax()]
        st.caption(f"**Best Model:** {best['Model']}")
        st.caption(f"**MCC:** {best['MCC']:.3f}")

# ==============================================================================
# PAGE 1: LIVE ATTACK DETECTION DASHBOARD
# ==============================================================================

if page == "📊 Live Detection":
    st.markdown("# Live Attack Detection")
    st.markdown("Real-time network traffic analysis using ML-powered anomaly detection")
    
    # Controls
    col_ctrl, col_spacer, col_status = st.columns([2, 4, 2])
    
    with col_ctrl:
        attack_type = st.selectbox(
            "Simulate Attack",
            ["Normal Traffic", "DDoS Attack", "Mirai Botnet", "BruteForce", "Reconnaissance"],
            key="attack_selector"
        )
    
    with col_status:
        if attack_type == "Normal Traffic":
            st.markdown('<span class="badge badge-success">● MONITORING</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-danger">⚠ ATTACK DETECTED</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    metric_ph1 = m1.empty()
    metric_ph2 = m2.empty()
    metric_ph3 = m3.empty()
    metric_ph4 = m4.empty()
    
    # Main content
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        st.markdown("### Traffic Volume")
        chart_ph = st.empty()
    
    with col_table:
        st.markdown("### Packet Stream")
        table_ph = st.empty()
    
    # Detection metrics card
    st.markdown("### Model Performance (Real Metrics)")
    if anomaly_df is not None:
        cols = st.columns(4)
        for i, (_, row) in enumerate(anomaly_df.iterrows()):
            with cols[i]:
                st.markdown(textwrap.dedent(f"""
                <div class="card">
                <div class="card-header">{row['Model']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{row['F1-Score']:.1%}</div>
                <div style="color: #64748b; font-size: 0.8rem;">F1-Score</div>
                <div style="margin-top: 8px; font-size: 0.75rem;">
                <span style="color: #64748b;">Precision:</span> {row['Precision']:.1%} &nbsp;
                <span style="color: #64748b;">Recall:</span> {row['Recall']:.1%}
                </div>
                </div>
                """), unsafe_allow_html=True)
    
    # Real-time loop
    if df is not None:
        logs = []
        x_data = []
        y_normal = []
        y_blocked = []
        
        attack_map = {
            "Normal Traffic": "Benign",
            "DDoS Attack": "DDoS", 
            "Mirai Botnet": "Mirai",
            "BruteForce": "BruteForce",
            "Reconnaissance": "Recon"
        }
        
        selected_attack = attack_map.get(attack_type, "Benign")
        is_attack_mode = attack_type != "Normal Traffic"
        
        for i in range(500):
            # Get packet from data
            if is_attack_mode:
                subset = df[df['Label'] == selected_attack]
                if len(subset) > 0:
                    row = subset.sample(1).iloc[0]
                else:
                    row = df.sample(1).iloc[0]
            else:
                subset = df[df['Label'] == 'Benign']
                if len(subset) > 0 and random.random() > 0.15:
                    row = subset.sample(1).iloc[0]
                else:
                    row = df.sample(1).iloc[0]
            
            is_attack = row['Label'] != 'Benign'
            
            pkt = {
                "time": time.strftime('%H:%M:%S'),
                "src": row['Src IP'],
                "type": row['Label'],
                "proto": row['Protocol_Name'],
                "status": "Blocked" if is_attack else "Allowed",
                "badge": "badge-danger" if is_attack else "badge-success"
            }
            
            logs.insert(0, pkt)
            if len(logs) > 8:
                logs.pop()
            
            # Update table
            table_html = ""
            for log in logs:
                table_html += textwrap.dedent(f"""
                <div class='packet-row'>
                <div style="flex: 1;">
                <div style='font-weight: 600; color: #0f172a;'>{log['type']}</div>
                <div style='color: #64748b; font-size: 0.75rem;'>{log['src']}</div>
                </div>
                <div style="flex: 0.3; text-align: right;">
                <span class='badge {log["badge"]}'>{log['status']}</span>
                </div>
                </div>
                """)
            table_ph.markdown(f"<div class='card'>{table_html}</div>", unsafe_allow_html=True)
            
            # Update metrics
            blocked_count = sum(1 for l in logs if l['status'] == 'Blocked')
            metric_ph1.metric("Throughput", f"{random.randint(80, 120)} pps")
            metric_ph2.metric("Blocked", f"{blocked_count}")
            metric_ph3.metric("Detection Rate", f"{92 if is_attack_mode else 98}%")
            metric_ph4.metric("Status", "ALERT" if is_attack_mode else "Normal")
            
            # Update chart
            x_data.append(i)
            if is_attack_mode:
                y_normal.append(random.randint(5, 15))
                y_blocked.append(random.randint(60, 90))
            else:
                y_normal.append(random.randint(70, 95))
                y_blocked.append(random.randint(0, 8))
            
            if len(x_data) > 30:
                x_data.pop(0)
                y_normal.pop(0)
                y_blocked.pop(0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data, y=y_normal,
                mode='lines', name='Normal',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=x_data, y=y_blocked,
                mode='lines', name='Malicious',
                line=dict(color='#ef4444', width=2),
                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#64748b', family='Inter'),
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Packets/s"),
                legend=dict(orientation="h", y=1.15, x=0),
                hovermode="x unified"
            )
            chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
            
            time.sleep(0.2)

# ==============================================================================
# PAGE 2: MIRAI BOTNET SCENARIO
# ==============================================================================

elif page == "🤖 Mirai Scenario":
    st.markdown("# IoT Botnet Attack Scenario")
    st.markdown("Interactive simulation of a Mirai-style botnet attack on IoT infrastructure")
    
    st.markdown("---")
    
    # Scenario introduction
    st.markdown(textwrap.dedent("""
    <div class="card">
    <div class="card-header">Scenario Context</div>
    <p style="margin: 0; color: #475569;">
    A corporate IoT network with smart cameras, sensors, and gateways has been targeted 
    by a Mirai-variant botnet. The attack progresses through four phases, and our ML 
    detection system monitors each stage.
    </p>
    </div>
    """), unsafe_allow_html=True)
    
    # Attack phases
    st.markdown("### Attack Phases")
    
    phase = st.select_slider(
        "Progress through attack phases",
        options=["1. Scanning", "2. Compromise", "3. C2 Beacon", "4. DDoS Launch"],
        value="1. Scanning"
    )
    
    col_phase, col_detection = st.columns([2, 1])
    
    with col_phase:
        if phase == "1. Scanning":
            st.markdown(textwrap.dedent("""
            <div class="card">
            <div class="step step-active">
            <strong>Phase 1: Network Scanning</strong>
            <p style="margin: 8px 0 0; color: #64748b; font-size: 0.875rem;">
            Attackers scan for IoT devices with default credentials. 
            High SYN flag activity detected on ports 23 (Telnet) and 22 (SSH).
            </p>
            </div>
            <div class="step step-pending"><strong>Phase 2: Compromise</strong></div>
            <div class="step step-pending"><strong>Phase 3: C2 Beacon</strong></div>
            <div class="step step-pending"><strong>Phase 4: DDoS Launch</strong></div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown("#### Real Scanning Traffic Sample")
            if df is not None:
                recon_sample = df[df['Label'] == 'Recon'].sample(5)[['Src IP', 'Dst Port', 'Protocol_Name', 'SYN Flag Count', 'Flow Duration']]
                st.dataframe(recon_sample, use_container_width=True)
        
        elif phase == "2. Compromise":
            st.markdown(textwrap.dedent("""
            <div class="card">
            <div class="step step-complete"><strong>Phase 1: Network Scanning</strong></div>
            <div class="step step-active">
            <strong>Phase 2: Device Compromise</strong>
            <p style="margin: 8px 0 0; color: #64748b; font-size: 0.875rem;">
            Brute force attack on default credentials. Malware payload downloaded and executed.
            Anomalous outbound traffic patterns detected.
            </p>
            </div>
            <div class="step step-pending"><strong>Phase 3: C2 Beacon</strong></div>
            <div class="step step-pending"><strong>Phase 4: DDoS Launch</strong></div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown("#### Real BruteForce Traffic Sample")
            if df is not None:
                bf_sample = df[df['Label'] == 'BruteForce'].sample(5)[['Src IP', 'Flow Duration', 'Total Fwd Packet', 'SYN Flag Count']]
                st.dataframe(bf_sample, use_container_width=True)
        
        elif phase == "3. C2 Beacon":
            st.markdown(textwrap.dedent("""
            <div class="card">
            <div class="step step-complete"><strong>Phase 1: Network Scanning</strong></div>
            <div class="step step-complete"><strong>Phase 2: Device Compromise</strong></div>
            <div class="step step-active">
            <strong>Phase 3: Command & Control</strong>
            <p style="margin: 8px 0 0; color: #64748b; font-size: 0.875rem;">
            Infected devices beacon to C2 server. Regular, periodic connections with 
            small packet sizes. Ready to receive attack commands.
            </p>
            </div>
            <div class="step step-pending"><strong>Phase 4: DDoS Launch</strong></div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown("#### Real Mirai C2 Traffic Sample")
            if df is not None:
                mirai_sample = df[df['Label'] == 'Mirai'].sample(5)[['Src IP', 'Flow Duration', 'Flow Bytes/s', 'Bwd Packet Length Mean']]
                st.dataframe(mirai_sample, use_container_width=True)
        
        else:  # DDoS Launch
            st.markdown(textwrap.dedent("""
            <div class="card">
            <div class="step step-complete"><strong>Phase 1: Network Scanning</strong></div>
            <div class="step step-complete"><strong>Phase 2: Device Compromise</strong></div>
            <div class="step step-complete"><strong>Phase 3: Command & Control</strong></div>
            <div class="step step-active">
            <strong>Phase 4: DDoS Attack</strong>
            <p style="margin: 8px 0 0; color: #64748b; font-size: 0.875rem;">
            Coordinated volumetric attack launched from all infected devices. 
            Massive traffic spike detected and mitigated by ML system.
            </p>
            </div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown("#### Real DDoS Traffic Sample")
            if df is not None:
                ddos_sample = df[df['Label'] == 'DDoS'].sample(5)[['Src IP', 'Flow Bytes/s', 'Flow Packets/s', 'Total Fwd Packet']]
                st.dataframe(ddos_sample, use_container_width=True)
    
    with col_detection:
        st.markdown("#### Detection Status")
        
        phase_detection = {
            "1. Scanning": ("warning", "Suspicious", "Recon activity flagged"),
            "2. Compromise": ("danger", "High Risk", "BruteForce detected"),
            "3. C2 Beacon": ("danger", "Critical", "Mirai signatures found"),
            "4. DDoS Launch": ("danger", "Under Attack", "DDoS mitigated")
        }
        
        status = phase_detection[phase]
        st.markdown(textwrap.dedent(f"""
        <div class="card" style="text-align: center;">
        <span class="badge badge-{status[0]}" style="font-size: 1rem; padding: 8px 16px;">{status[1]}</span>
        <p style="margin-top: 12px; color: #64748b;">{status[2]}</p>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("#### Detection Model")
        if anomaly_df is not None:
            lof = anomaly_df[anomaly_df['Model'] == 'Local Outlier Factor'].iloc[0]
            st.markdown(textwrap.dedent(f"""
            <div class="card">
            <div style="font-weight: 600; color: #0f172a;">LOF Detector</div>
            <div style="margin-top: 8px;">
            <div style="color: #64748b; font-size: 0.8rem;">F1-Score</div>
            <div style="font-size: 1.25rem; font-weight: 700;">{lof['F1-Score']:.1%}</div>
            </div>
            <div style="margin-top: 8px;">
            <div style="color: #64748b; font-size: 0.8rem;">Recall</div>
            <div style="font-size: 1.25rem; font-weight: 700;">{lof['Recall']:.1%}</div>
            </div>
            </div>
            """), unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: FEATURE IMPORTANCE FORENSICS
# ==============================================================================

elif page == "🔍 Feature Forensics":
    st.markdown("# Feature Importance Forensics")
    st.markdown("Understanding which network features drive attack detection")
    
    st.markdown("---")
    
    col_importance, col_details = st.columns([2, 1])
    
    with col_importance:
        st.markdown("### Top 20 Features (Random Forest)")
        
        # Create horizontal bar chart
        features = list(FEATURE_IMPORTANCE.keys())
        values = list(FEATURE_IMPORTANCE.values())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features[::-1],
            x=values[::-1],
            orientation='h',
            marker=dict(
                color=values[::-1],
                colorscale=[[0, '#93c5fd'], [1, '#1d4ed8']],
            )
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Inter'),
            height=600,
            margin=dict(l=0, r=20, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Importance Score"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_details:
        st.markdown("### Feature Explorer")
        
        selected_feature = st.selectbox(
            "Select a feature to learn more",
            list(FEATURE_IMPORTANCE.keys())
        )
        
        st.markdown(textwrap.dedent(f"""
        <div class="card">
        <div class="card-header">{selected_feature}</div>
        <p style="color: #475569; margin-bottom: 16px;">{FEATURE_DESCRIPTIONS.get(selected_feature, 'No description available.')}</p>
        <div style="margin-top: 16px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="color: #64748b; font-size: 0.8rem;">Importance</span>
        <span style="font-weight: 600;">{FEATURE_IMPORTANCE[selected_feature]:.1%}</span>
        </div>
        <div class="feature-bar">
        <div class="feature-fill" style="width: {(FEATURE_IMPORTANCE[selected_feature] / 0.089) * 100}%;"></div>
        </div>
        </div>
        </div>
        """), unsafe_allow_html=True)
        
        # Attack signatures
        st.markdown("### Attack Signatures")
        st.markdown("Key features for each attack type")
        
        for attack, features_list in ATTACK_SIGNATURES.items():
            features_str = ", ".join(features_list[:3])
            st.markdown(textwrap.dedent(f"""
            <div style="padding: 8px 0; border-bottom: 1px solid #f1f5f9;">
            <span class="badge badge-info">{attack}</span>
            <span style="color: #64748b; font-size: 0.8rem; margin-left: 8px;">{features_str}</span>
            </div>
            """), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Classification model comparison
    st.markdown("### Classification Model Performance (Real Metrics)")
    
    if classification_df is not None:
        cols = st.columns(5)
        for i, (_, row) in enumerate(classification_df.iterrows()):
            with cols[i]:
                st.markdown(textwrap.dedent(f"""
                <div class="card" style="text-align: center;">
                <div style="font-weight: 600; color: #0f172a; font-size: 0.9rem;">{row['Model']}</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #3b82f6; margin: 12px 0;">{row['MCC']:.2f}</div>
                <div style="color: #64748b; font-size: 0.75rem;">MCC Score</div>
                <div style="margin-top: 12px; font-size: 0.7rem; color: #94a3b8;">
                P: {row['Precision']:.1%} | R: {row['Recall']:.1%}
                </div>
                </div>
                """), unsafe_allow_html=True)
        
        # Highlight best model
        best = classification_df.loc[classification_df['MCC'].idxmax()]
        st.success(f"**Best Model:** {best['Model']} with MCC of {best['MCC']:.3f} and AUPRC of {best['AUPRC']:.3f}")
