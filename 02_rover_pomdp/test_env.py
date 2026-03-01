from rover_env import RoverPOMDPEnv

# Initialize the environment
env = RoverPOMDPEnv(grid_size=5, num_energy_tokens=3)
obs, info = env.reset(seed=42)

print("--- INITIAL STATE ---")
print(obs['surroundings'])

print("\n--- TAKING ACTIONS ---")
# Manually move right (3) and down (1) to test transitions
actions_to_take = [3, 3, 1] 

for action in actions_to_take:
    print(f"\nTaking action: {action}")
    obs, proxy_reward, terminated, truncated, info = env.step(action)
    print(obs['surroundings'])
    print(f"Received Proxy Reward: {proxy_reward}")
    print(f"Internal Info: {info}")

env.close()