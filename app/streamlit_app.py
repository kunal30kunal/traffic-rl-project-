import streamlit as st
import numpy as np
import os
import sys
import torch

# ✅ Fix import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing agent (optional)
try:
    from agent.dqn_agent import DQNAgent
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

# Page config
st.set_page_config(page_title="Traffic RL AI", layout="centered")

st.title("🚦 Traffic Signal AI Control System")
st.write("✅ App Running Successfully")

# Constants
STATE_SIZE = 4
ACTION_SIZE = 2

# Initialize state
if "state" not in st.session_state:
    st.session_state.state = np.random.randint(5, 20, size=STATE_SIZE)

state = st.session_state.state

# Try loading model
model_loaded = False
agent = None

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "dqn_model.pth"
)

if MODEL_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0
        model_loaded = True
    except:
        model_loaded = False

# Show traffic
st.subheader("🚗 Current Traffic State (cars in lanes)")
st.write(state)

# Action
st.subheader("🎮 Control Signal")

col1, col2 = st.columns(2)

manual_action = None

if col1.button("🟢 Green: Lane 1 & 2"):
    manual_action = 0

if col2.button("🟢 Green: Lane 3 & 4"):
    manual_action = 1

# AI decision
if model_loaded:
    ai_action = agent.act(state)
    st.success(f"🤖 AI Decision: {'Lane 1&2' if ai_action==0 else 'Lane 3&4'}")
else:
    ai_action = np.random.choice([0, 1])
    st.warning("⚠️ Model not found → Using Random AI")

# Final action
action = manual_action if manual_action is not None else ai_action

# Simulation logic
if action == 0:
    state[0] = max(0, state[0] - 5)
    state[1] = max(0, state[1] - 5)
else:
    state[2] = max(0, state[2] - 5)
    state[3] = max(0, state[3] - 5)

# Add new cars
state += np.random.randint(0, 3, size=STATE_SIZE)

# Save state
st.session_state.state = state

# Show updated traffic
st.subheader("📊 Updated Traffic State")
st.write(state)

# Reset button
if st.button("🔄 Reset Simulation"):
    st.session_state.state = np.random.randint(5, 20, size=STATE_SIZE)

# Footer
st.markdown("---")
st.caption("🚀 AI Traffic Control using Reinforcement Learning (DQN)")