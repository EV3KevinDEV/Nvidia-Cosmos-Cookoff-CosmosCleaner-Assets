import streamlit as st
import requests
import time
import io
import plotly.graph_objects as go
import numpy as np

BRIDGE_URL = "http://127.0.0.1:5765"

st.set_page_config(page_title="CosmosCleanerBot Dashboard", layout="wide")

def get_state():
    try:
        resp = requests.get(f"{BRIDGE_URL}/state", timeout=0.1)
        return resp.json()
    except:
        return None

def post_cmd(cmd):
    try:
        requests.post(f"{BRIDGE_URL}/cmd", json=cmd, timeout=0.1)
    except:
        pass

st.title("🤖 CosmosCleanerBot Dashboard")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Camera Feed")
    video_placeholder = st.empty()
    
    st.divider()
    
    st.subheader("🕹️ Controls")
    mode = st.radio("Drive Mode", ["Manual", "Auto (Square)"], index=0)
    if mode == "Auto (Square)":
        post_cmd({"mode": "auto"})
    else:
        post_cmd({"mode": "manual"})

    c1, c2 = st.columns(2)
    lin = c1.slider("Linear Vel", -0.5, 0.5, 0.0, 0.05)
    ang = c2.slider("Angular Vel", -2.0, 2.0, 0.0, 0.1)
    
    if st.button("Apply Manual Vel", use_container_width=True):
        post_cmd({"linear": lin, "angular": ang})
    
    if st.button("STOP", use_container_width=True, type="primary"):
        post_cmd({"linear": 0.0, "angular": 0.0, "mode": "manual"})

with col2:
    st.subheader("📊 Telemetry")
    telem_placeholder = st.empty()
    
    st.divider()
    
    st.subheader("🗺️ Trajectory")
    traj_placeholder = st.empty()

# Refresh loop
while True:
    state = get_state()
    if state:
        # Update camera
        try:
            img_resp = requests.get(f"{BRIDGE_URL}/frame", timeout=0.1)
            if img_resp.status_code == 200:
                video_placeholder.image(img_resp.content, caption="Robot View", use_container_width=True)
        except:
            pass
            
        # Update Telemetry
        with telem_placeholder.container():
            st.write(f"**Mode:** {state['mode']}")
            st.write(f"**Position:** X: {state['pos'][0]:.3f}, Y: {state['pos'][1]:.3f}")
            st.write(f"**Yaw:** {state['yaw']:.1f}°")
            st.write(f"**Cmd:** v: {state['v_cmd']:.2f}, ω: {state['w_cmd']:.2f}")

        # Update Trajectory
        fig = go.Figure()
        # Path
        fig.add_trace(go.Scatter(x=state['traj_x'], y=state['traj_y'], mode='lines', name='Path', line=dict(color='blue')))
        # Waypoints
        if state['waypoints']:
            wps = list(zip(*state['waypoints']))
            fig.add_trace(go.Scatter(x=wps[0], y=wps[1], mode='markers', name='Waypoints', marker=dict(color='green', size=10)))
        # Robot
        fig.add_trace(go.Scatter(x=[state['pos'][0]], y=[state['pos'][1]], mode='markers', name='Robot', marker=dict(color='red', size=12)))
        
        fig.update_layout(
            xaxis_title="X (m)", yaxis_title="Y (m)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            yaxis_scaleanchor="x"
        )
        traj_placeholder.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Waiting for bridge at http://127.0.0.1:5765 ...")
    
    time.sleep(0.5)
