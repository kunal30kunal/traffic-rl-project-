import streamlit as st
import numpy as np
import os
import sys
import torch

# Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.traffic_env import TrafficEnv

# Try importing agent (optional)
try:
    from agent.dqn_agent import DQNAgent
    MODEL_CODE_AVAILABLE = True
except:
    MODEL_CODE_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Traffic RL AI", layout="centered")

st.title("🚦 Traffic Signal AI Control System")
st.write("✅ App Running Successfully")

# ---------------- MODEL PATH ----------------
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "dqn_model.pth")
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if MODEL_CODE_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            agent = DQNAgent(state_size=4, action_size=2)
            agent.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )
            agent.model.eval()
            return agent
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None
    return None

agent = load_model()

# ---------------- ENV ----------------
env = TrafficEnv()
state, _ = env.reset()

# ---------------- UI ----------------
st.subheader("🚗 Current Traffic State (cars in lanes)")
st.table(state.reshape(-1, 1))

# ---------------- ACTION ----------------
if agent:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(agent.model(state_tensor)).item()
    st.success("✅ Using Trained AI Model")
else:
    action = np.random.choice([0, 1])
    st.warning("⚠️ Model not found → Using Random AI")

# ---------------- SIGNAL DISPLAY ----------------
st.subheader("🎮 Control Signal")

col1, col2 = st.columns(2)

with col1:
    if action == 0:
        st.success("🟢 Green: Lane 1 & 2")
    else:
        st.write("🔴 Red: Lane 1 & 2")

with col2:
    if action == 1:
        st.success("🟢 Green: Lane 3 & 4")
    else:
        st.write("🔴 Red: Lane 3 & 4")

# ---------------- STEP ENV ----------------
next_state, reward, _, _, _ = env.step(action)

st.subheader("📊 Updated Traffic State")
st.table(next_state.reshape(-1, 1))

st.subheader("🏆 Reward")
st.write(reward)

# ---------------- DEBUG ----------------
with st.expander("🔍 Debug Info"):
    st.write("Model Path:", MODEL_PATH)
    st.write("Model Exists:", os.path.exists(MODEL_PATH))
    st.write("Model Code Available:", MODEL_CODE_AVAILABLE)