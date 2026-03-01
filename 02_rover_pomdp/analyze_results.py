import pandas as pd
import os

def analyze_and_compare():
    # Define your file paths
    zero_shot_file = "rover_experiment_results.csv"
    cot_file = "rover_cot_experiment_results.csv"
    
    dataframes = []
    
    # Load Zero-Shot Data
    if os.path.exists(zero_shot_file):
        df_zs = pd.read_csv(zero_shot_file)
        df_zs['Method'] = 'Zero-Shot'
        dataframes.append(df_zs)
        
    # Load Chain-of-Thought Data
    if os.path.exists(cot_file):
        df_cot = pd.read_csv(cot_file)
        df_cot['Method'] = 'Chain-of-Thought'
        dataframes.append(df_cot)
        
    if not dataframes:
        print("No CSV files found. Please run the experiment first.")
        return
        
    # Combine the datasets
    df_all = pd.concat(dataframes, ignore_index=True)
    
    # Calculate summary statistics grouped by Method and Condition
    summary = df_all.groupby(['Method', 'Condition']).agg(
        Goal_Reach_Rate_Pct=('Reached_Goal', lambda x: x.mean() * 100),
        True_Return_Mean=('True_Return', 'mean'),
        Proxy_Return_Mean=('Observed_Return (Proxy)', 'mean'),
        Misalignment_Gap_Mean=('Misalignment_Gap', 'mean'),
        Avg_Steps=('Steps_Taken', 'mean')
    ).round(2).reset_index()

    # Reorder conditions for logical academic presentation
    condition_order = ['Goal_Only', 'Reward_Only', 'Both']
    summary['Condition'] = pd.Categorical(summary['Condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values(['Method', 'Condition'])

    # Print the final Markdown table for the paper
    print("### Comparative Results: Zero-Shot vs. Chain-of-Thought\n")
    print(summary.to_markdown(index=False))

    # Quick check for Qualitative Analysis
    if 'Reasoning_Log' in df_all.columns:
        print("\n" + "-"*50)
        print("### Qualitative Analysis Primer")
        df_cot_only = df_all[df_all['Method'] == 'Chain-of-Thought']
        failed_episodes = len(df_cot_only[df_cot_only['Reached_Goal'] == 0])
        print(f"Total CoT failures to investigate: {failed_episodes}")
        print("Next Step: Open 'rover_cot_experiment_results.csv' in your IDE.")
        print("Read the 'Reasoning_Log' for the failed episodes to build your Failure Mode Taxonomy.")

if __name__ == "__main__":
    analyze_and_compare()