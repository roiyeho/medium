import gymnasium as gym

env = gym.make('Taxi-v3', render_mode='human')

observation, info = env.reset()
print(observation)

# Sample a random action from all valid actions
action = env.action_space.sample()
observation, reward, terminated, truncated, _ = env.step(action)
print('Observation:', observation)
print('Reward:', reward)
print('Terminated:', terminated)
print('Truncated:', truncated)

# Basic interaction loop
for i in range(25):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(action)
    
env.close()