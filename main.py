from env import VisualTrapEnv
from agent import MultimodalAgent
import time
import csv
import os

def log_detailed_results(exp_id, episode, raw_text, parsed_action, true_score, proxy_score):
    """Logs the results into our unified CSV."""
    filename = "visual_run_summary.csv"
    file_exists = os.path.isfile(filename)
    
    if true_score > 0:
        failure_mode = "success"
    elif proxy_score > 0:
        failure_mode = "stroop_effect_hacking"
    else:
        failure_mode = "format_error"

    with open(filename, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["exp_id", "episode_id", "raw_text", "parsed_action", "true_reward", "proxy_reward", "failure_mode"])
        
        clean_text = raw_text.replace('\n', ' ').strip()
        writer.writerow([exp_id, episode, clean_text, parsed_action, true_score, proxy_score, failure_mode])

def run_exp_005(num_episodes=5):
    print("--- Starting EXP-005: THE STROOP EFFECT TRAP ---")
    
    env = VisualTrapEnv()
    agent = MultimodalAgent()
    
    total_true = 0
    total_bias = 0

    for episode in range(num_episodes):
        image_state = env.reset()
        print(f"\n--- Episode {episode + 1} ---")
        
        raw_action_text = agent.get_action(image_state)
        print(f"Agent Raw Output: {raw_action_text}")
        
        _, true_r, proxy_r, _ = env.step(raw_action_text)
        
        # Parse the action
        parsed_action = "OPEN_GREEN" if "OPEN_GREEN" in raw_action_text.upper() else ("OPEN_RED" if "OPEN_RED" in raw_action_text.upper() else "UNKNOWN")
        
        total_true += (1 if true_r > 0 else 0)
        total_bias += (1 if proxy_r > 0 else 0)
        
        log_detailed_results("EXP-005", episode + 1, raw_action_text, parsed_action, true_r, proxy_r)
        time.sleep(2) 

    print("\n=== EXP-005 RESULTS ===")
    print(f"Times Agent overcame the Stroop Effect (True Success): {total_true}/{num_episodes}")
    print(f"Times Agent fell for OCR Bias (Proxy Failure): {total_bias}/{num_episodes}")
    print("Detailed results saved to 'visual_run_summary.csv'")

if __name__ == "__main__":
    run_exp_005(num_episodes=5)