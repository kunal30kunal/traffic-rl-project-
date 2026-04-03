import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Traffic RL Control", layout="centered")

st.title("🚦 Traffic Signal Optimization using RL")
st.markdown("Deep Q-Network based smart traffic control system")

# ===== INIT ENV =====
env = TrafficEnv()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ===== LOAD MODEL =====
@st.cache_resource
def load_agent():
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load_model("models/dqn_model.pth")
        agent.epsilon = 0.0   # 🔥 FIX: no randomness
        return agent, True
    except:
        return agent, False

agent, model_loaded = load_agent()

if model_loaded:
    st.success("✅ Trained model loaded (AI mode)")
else:
    st.warning("⚠️ No trained model found (Random mode)")

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Controls")

steps = st.sidebar.slider("Simulation Steps", 10, 200, 50)
speed = st.sidebar.slider("Speed (seconds)", 0.05, 1.0, 0.2)

mode = st.sidebar.selectbox(
    "Mode",
    ["AI Control", "Random Control"]
)

run_btn = st.sidebar.button("▶ Run Simulation")

# ===== MAIN =====
if run_btn:
    state, _ = env.reset()

    total_reward = 0
    rewards = []

    chart = st.line_chart()
    state_box = st.empty()

    for step in range(steps):

        # ===== ACTION =====
        if mode == "AI Control" and model_loaded:
            action = agent.select_action(state)
        else:
            action = np.random.randint(0, action_size)

        # ===== STEP =====
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        rewards.append(total_reward)

        # ===== DISPLAY STATE =====
        state_text = "### 🚗 Current Traffic State\n"
        for i in range(len(next_state)):
            state_text += f"Lane {i+1}: {int(next_state[i])} cars\n"

        state_text += f"\n🟢 Green Signal: **Lane {action+1}**"

        state_box.markdown(state_text)

        # ===== UPDATE GRAPH =====
        chart.add_rows([[total_reward]])

        state = next_state

        if done:
            break

        time.sleep(speed)

    # ===== FINAL OUTPUT =====
    st.success(f"🎯 Total Reward: {total_reward}")

    # ===== PERFORMANCE MESSAGE =====
    if total_reward > -800:
        st.success("🔥 Excellent Traffic Control!")
    elif total_reward > -1500:
        st.info("👍 Decent Performance")
    else:
        st.error("⚠️ Poor Performance (Try training more)")
