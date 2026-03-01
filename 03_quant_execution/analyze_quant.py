import pandas as pd
import os

def analyze_quant_data(csv_filepath="quant_cot_experiment_results.csv"):
    if not os.path.exists(csv_filepath):
        print(f"File not found: {csv_filepath}")
        return

    df = pd.read_csv(csv_filepath)

    summary = df.groupby('Condition').agg(
        Order_Completion_Rate=('Order_Completed', lambda x: x.mean() * 100),
        Avg_Final_Price=('Final_Market_Price', 'mean'),
        True_Return_Mean=('True_Return', 'mean'),
        True_Return_Var=('True_Return', 'var'),
        Proxy_Return_Mean=('Observed_Return (Proxy)', 'mean'),
        Proxy_Return_Var=('Observed_Return (Proxy)', 'var'),
        Misalignment_Gap_Mean=('Misalignment_Gap', 'mean'),
        Misalignment_Gap_Var=('Misalignment_Gap', 'var')
    ).round(2).reset_index()

    condition_order = ['Goal_Only', 'Reward_Only', 'Both']
    summary['Condition'] = pd.Categorical(summary['Condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values('Condition')

    print("### Empirical Results: Algorithmic Trade Execution (POMDP)\n")
    print(summary.to_markdown(index=False))

if __name__ == "__main__":
    analyze_quant_data()