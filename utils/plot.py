import matplotlib.pyplot as plt


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Performance (Traffic RL)")
    plt.show()