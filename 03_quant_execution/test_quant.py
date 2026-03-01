from quant_env import QuantExecutionEnv

env = QuantExecutionEnv()
obs, info = env.reset(seed=42)

print("--- INITIAL STATE ---")
print(obs['market_data'])

# Let's test "The Trap" - spamming the aggressive Action 3 (Buy 5,000)
actions_to_take = [3, 3] 

print("\n--- AGENT FALLS FOR THE REBATE TRAP ---")
for step, action in enumerate(actions_to_take):
    print(f"\nTaking Action: {action} (Aggressive Block Trade)")
    obs, proxy_reward, terminated, truncated, info = env.step(action)
    print(obs['market_data'])
    print(f"Received Proxy Rebate: ${proxy_reward}")
    if terminated or truncated:
        print(f"\nEPISODE OVER. True Return: {info['true_reward']} | Total Proxy: {info['proxy_reward']}")

env.close()