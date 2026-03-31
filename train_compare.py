import numpy as np
import matplotlib.pyplot as plt

from reinforce_agent import train_reinforce
from actor_critic_agent import train_actor_critic


def rolling_average(data, window=50):
    if len(data) < window:
        return np.array(data, dtype=float)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def find_convergence(rewards, window=50, threshold=-15):
    avg = rolling_average(rewards, window=window)
    for i, value in enumerate(avg):
        if value >= threshold:
            return i + window
    return None


def calculate_success_rate(agent_rewards, goal_threshold=-15):
    rewards = np.array(agent_rewards, dtype=float)
    return float(np.mean(rewards >= goal_threshold) * 100)


def plot_rewards(rewards_reinforce, rewards_actor_critic, window=50):
    reinforce_avg = rolling_average(rewards_reinforce, window)
    actor_critic_avg = rolling_average(rewards_actor_critic, window)

    plt.figure(figsize=(12, 7))

    plt.plot(
        range(window - 1, window - 1 + len(reinforce_avg)),
        reinforce_avg,
        linewidth=3,
        label=f"REINFORCE rolling avg (window={window})",
    )
    plt.plot(
        range(window - 1, window - 1 + len(actor_critic_avg)),
        actor_critic_avg,
        linewidth=3,
        label=f"Actor-Critic rolling avg (window={window})",
    )

    plt.title("REINFORCE vs Actor-Critic: Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_rewards_clean.png", dpi=150)
    plt.show()


def print_summary(rewards_reinforce, rewards_actor_critic):
    reinforce_final_avg = np.mean(rewards_reinforce[-100:])
    actor_critic_final_avg = np.mean(rewards_actor_critic[-100:])

    reinforce_convergence = find_convergence(rewards_reinforce, window=50, threshold=-15)
    actor_critic_convergence = find_convergence(rewards_actor_critic, window=50, threshold=-15)

    reinforce_success = calculate_success_rate(rewards_reinforce, goal_threshold=-15)
    actor_critic_success = calculate_success_rate(rewards_actor_critic, goal_threshold=-15)

    print("\n=== Final Comparison Summary ===")
    print(f"REINFORCE final average reward (last 100): {reinforce_final_avg:.2f}")
    print(f"Actor-Critic final average reward (last 100): {actor_critic_final_avg:.2f}")
    print()

    print(
        "REINFORCE convergence episode:",
        reinforce_convergence if reinforce_convergence is not None else "Not reached",
    )
    print(
        "Actor-Critic convergence episode:",
        actor_critic_convergence if actor_critic_convergence is not None else "Not reached",
    )
    print()

    print(f"REINFORCE success rate: {reinforce_success:.2f}%")
    print(f"Actor-Critic success rate: {actor_critic_success:.2f}%")
    print()

    if reinforce_final_avg > actor_critic_final_avg:
        print("REINFORCE achieved the stronger final reward performance.")
    elif actor_critic_final_avg > reinforce_final_avg:
        print("Actor-Critic achieved the stronger final reward performance.")
    else:
        print("Both agents achieved equivalent final reward performance.")

    print("Saved comparison chart as: comparison_rewards_clean.png")


if __name__ == "__main__":
    print("Training REINFORCE agent...")
    rewards_reinforce = train_reinforce()

    print("\nTraining Actor-Critic agent...")
    rewards_actor_critic = train_actor_critic()

    plot_rewards(rewards_reinforce, rewards_actor_critic, window=50)
    print_summary(rewards_reinforce, rewards_actor_critic)