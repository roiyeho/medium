import gymnasium as gym
from tqdm import tqdm
from q_learning_agent import QLearningAgent

# Hyperparameters
n_episodes = 1000
learning_rate = 0.5
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)
final_epsilon = 0.1

env = gym.make('Taxi-v3')

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

def train_agent(agent):
    pbar = tqdm(range(n_episodes))
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        n_steps = 0
        total_reward = 0

        # Run one episode
        while not done:        
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs
            n_steps += 1

        pbar.set_description(f'Episode Length: {n_steps}, Reward: {total_reward}')
        agent.decay_epsilon()

def show_policy(agent: QLearningAgent, n_episodes=10):
    env = gym.make('Taxi-v3', render_mode='human')
    agent.epsilon = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:        
            action = agent.get_action(obs)
            # set epsilon to 0?
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
        
train_agent(agent)
show_policy(agent)