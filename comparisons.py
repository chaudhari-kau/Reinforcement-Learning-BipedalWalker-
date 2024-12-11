import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import savgol_filter

def load_training_logs(env_type, log_dir="logs"):
    """Load training logs for both SAC and TD3 for specific environment"""
    logs = {
        'sac': [],
        'td3': []
    }
    
    for rl_type in ['sac', 'td3']:
        log_files = glob(os.path.join(log_dir, rl_type, f"*{env_type}*.json"))
        for file in log_files:
            try:
                with open(file, 'r') as f:
                    logs[rl_type].append(json.load(f))
                print(f"Loaded {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    return logs

def generate_score_distribution(env_type, logs, plot_dir):
    plt.figure(figsize=(12, 6))
    segments_data = []
    phase_labels = ['Early', 'Early-Mid', 'Late-Mid', 'Late']
    labels = []
    
    # First collect all data
    for rl_type in ['sac', 'td3']:
        for log in logs[rl_type]:
            total_episodes = len(log['scores'])
            segment_size = total_episodes // 4
            current_segments = []
            for i in range(4):  # Always create 4 segments
                start = i * segment_size
                end = (i + 1) * segment_size
                segment = log['scores'][start:end]
                current_segments.append(segment)
            segments_data.extend(current_segments)
            labels.extend([f'{rl_type.upper()}-{phase}' for phase in phase_labels])
    
    # Create the plot
    if segments_data:
        positions = list(range(1, len(segments_data) + 1))
        parts = plt.violinplot(segments_data, showmeans=True)
        
        # Customize appearance
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.6)
        
        plt.title(f'Score Distribution Comparison ({env_type})')
        plt.xlabel('Training Phase')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(positions, labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'score_distribution.png'), dpi=300)
    plt.close()

def generate_learning_efficiency(env_type, logs, plot_dir):
    plt.figure(figsize=(12, 6))
    colors = {'sac': '#2ecc71', 'td3': '#3498db'}
    line_styles = {'sac': '--', 'td3': '-'}
    
    for rl_type in ['sac', 'td3']:
        if logs[rl_type]:  # Check if we have data
            for log in logs[rl_type]:
                # Plot raw scores with low alpha
                plt.plot(log['episodes'], log['scores'], 
                        color=colors[rl_type], alpha=0.1)
                
                # Plot smoothed curve
                if len(log['episodes']) > 101:
                    smoothed_scores = savgol_filter(log['scores'], 101, 3)
                    plt.plot(log['episodes'], smoothed_scores,
                            color=colors[rl_type], 
                            linestyle=line_styles[rl_type],
                            alpha=0.8,
                            linewidth=2,
                            label=f"{rl_type.upper()}")
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Learning Efficiency Comparison ({env_type})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal line at 300 (solving threshold)
    plt.axhline(y=300, color='r', linestyle=':', alpha=0.5, label='Solving Threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'learning_efficiency.png'), dpi=300)
    plt.close()

def print_summary_statistics(env_type, logs):
    print(f"\nPerformance Summary for {env_type}:")
    for rl_type in ['sac', 'td3']:
        if logs[rl_type]:
            all_final_scores = []
            for log in logs[rl_type]:
                n_scores = min(100, len(log['scores']))
                all_final_scores.extend(log['scores'][-n_scores:])
                
            print(f"\n{rl_type.upper()}:")
            print(f"Final 100 episodes - Mean: {np.mean(all_final_scores):.2f}")
            print(f"Final 100 episodes - Std: {np.std(all_final_scores):.2f}")
            print(f"Best Score: {np.max(all_final_scores):.2f}")
        else:
            print(f"\n{rl_type.upper()}: No data available")

def generate_comparison_plots(env_type):
    """Generate comparative visualization plots between SAC and TD3"""
    logs = load_training_logs(env_type)
    
    # Create directory for plots
    plot_dir = os.path.join("results", "plots", "comparisons", env_type)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in: {plot_dir}")
    
    # Generate all plots
    generate_score_distribution(env_type, logs, plot_dir)
    generate_learning_efficiency(env_type, logs, plot_dir)
    
    # Print summary statistics
    print_summary_statistics(env_type, logs)

if __name__ == "__main__":
    print("Generating comparisons for classic environment...")
    generate_comparison_plots('classic')
    print("\nGenerating comparisons for hardcore environment...")
    generate_comparison_plots('hardcore')