from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
from config import EPISODES, MAX_STEPS
from utils.plot import plot_rewards


def train_model():
    env = TrafficEnv()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    rewards_list = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_list.append(total_reward)

        print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save model
    agent.save_model()

    # Plot graph
    plot_rewards(rewards_list)

    env.close()