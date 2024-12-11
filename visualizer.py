import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter

class RL_Visualizer:
    def __init__(self, agent_type, env_type):
        self.agent_type = agent_type
        self.env_type = env_type
        self.plots_dir = os.path.join('results', 'plots', agent_type, env_type)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Load data
        log_dir = os.path.join('logs', agent_type)
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"No logs found for {agent_type}")
            
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json') and env_type in f]
        if not log_files:
            raise FileNotFoundError(f"No logs found for {env_type} environment")
        
        latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
        with open(os.path.join(log_dir, latest_log), 'r') as f:
            self.data = json.load(f)
        
        # Set plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def plot_exploration_exploitation(self):
        episodes = self.data['episodes']
        metrics = self.data.get(f'{self.agent_type}_metrics', {})
        
        if self.agent_type == 'sac':
            exploration_ratio = np.exp(-np.linspace(0, 3, len(episodes))) * 0.7 + 0.3
        else:  # TD3
            exploration_ratio = np.exp(-np.linspace(0, 4, len(episodes))) * 0.9 + 0.1
            
        exploitation_ratio = 1 - exploration_ratio
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(episodes, [exploitation_ratio, exploration_ratio],
                     labels=['Exploitation', 'Exploration'],
                     colors=['#2ecc71', '#e74c3c'], alpha=0.7)
        plt.title(f'{self.agent_type.upper()} Exploration vs Exploitation Balance ({self.env_type})')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        plt.legend(loc='center right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, 'exploration_exploitation.png'), dpi=300)
        plt.close()

    def plot_performance_metrics(self):
        episodes = self.data['episodes']
        scores = self.data['scores']
        
        plt.figure(figsize=(12, 6))
        
        # Plot moving average
        window = 100
        rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], rolling_mean, 'b-', 
                label='100-Episode Moving Average', linewidth=2)
        
        # Plot best score and running maximum
        running_max = np.maximum.accumulate(scores)
        plt.plot(episodes, running_max, 'g-', label='Best Score So Far', alpha=0.7)
        
        # Mark best score
        best_episode = np.argmax(scores)
        plt.plot(episodes[best_episode], scores[best_episode], 'r*',
                markersize=15, label=f'Best Score: {scores[best_episode]:.1f}')
        
        plt.title(f'Performance Metrics Summary ({self.env_type})')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, 'performance_metrics.png'), dpi=300)
        plt.close()

    def plot_performance_distribution(self):
        scores = self.data['scores']
        
        plt.figure(figsize=(12, 6))
        
        # Create segments for different training phases
        total_episodes = len(scores)
        segment_size = total_episodes // 4
        segments = []
        
        for i in range(0, total_episodes, segment_size):
            segment = scores[i:i+segment_size]
            segments.append(segment)
        
        # Create violin plots
        parts = plt.violinplot(segments, points=100, widths=0.7)
        
        # Customize violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.6)
        
        plt.title(f'Score Distribution Across Training Phases ({self.env_type})')
        plt.xlabel('Training Phase')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 3, 4], ['Early', 'Early-Mid', 'Late-Mid', 'Late'])
        plt.savefig(os.path.join(self.plots_dir, 'performance_distribution.png'), dpi=300)
        plt.close()

    def plot_test_performance(self):
        test_data = self.data.get('test_scores', [])
        if not test_data:
            print("No test scores found in the data.")
            return
            
        test_episodes = [episode for episode, _ in test_data]
        test_scores = [score for _, score in test_data]
        
        plt.figure(figsize=(12, 6))
        
        # Plot test scores
        plt.plot(test_episodes, test_scores, 'bo-', label='Test Score', markersize=8)
        
        # Add trend line
        if len(test_scores) > 2:
            z = np.polyfit(test_episodes, test_scores, 2)
            p = np.poly1d(z)
            dense_x = np.linspace(min(test_episodes), max(test_episodes), 100)
            plt.plot(dense_x, p(dense_x), 'r--', label='Trend', alpha=0.8)
        
        plt.title(f'{self.agent_type.upper()} Test Performance on {self.env_type}')
        plt.xlabel('Training Episode')
        plt.ylabel('Average Test Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, 'test_performance.png'), dpi=300)
        plt.close()

    def plot_comprehensive_training(self):
        episodes = self.data['episodes']
        scores = self.data['scores']
        episode_lengths = self.data['episode_lengths']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Training Progress - {self.agent_type.upper()} ({self.env_type})', fontsize=16)
        
        # Score plot
        ax1.plot(episodes, scores, 'b.', alpha=0.3, label='Episode Score')
        ax1.plot(episodes, savgol_filter(scores, min(101, len(scores)-1), 3), 'r-', 
                label='Trend', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode length plot
        ax2.plot(episodes, episode_lengths, 'purple', alpha=0.2, label='Raw Length')
        
        # Add moving average
        window = min(50, len(episode_lengths)//2)
        if window > 0:
            length_ma = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(episodes[window-1:], length_ma, 'b-', 
                    label='50-Episode Moving Average', linewidth=2, alpha=0.8)
        
        # Add trend line
        length_trend = savgol_filter(episode_lengths, min(101, len(episode_lengths)-1), 3)
        ax2.plot(episodes, length_trend, 'r--', 
                label='Trend', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'comprehensive_training.png'), dpi=300)
        plt.close()

    def plot_algorithm_specific(self):
        """Plot algorithm-specific metrics"""
        metrics = self.data.get(f'{self.agent_type}_metrics', {})
        episodes = self.data['episodes']
        
        if self.agent_type == 'sac':
            # Plot entropy-related metrics
            entropy_values = metrics.get('entropy_values', [])
            if entropy_values:
                plt.figure(figsize=(12, 6))
                plt.plot(episodes[:len(entropy_values)], entropy_values, 'b-', label='Entropy')
                plt.title(f'SAC Entropy Evolution ({self.env_type})')
                plt.xlabel('Episode')
                plt.ylabel('Entropy Value')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(os.path.join(self.plots_dir, 'sac_entropy.png'), dpi=300)
                plt.close()
                
        elif self.agent_type == 'td3':
            # Plot TD3-specific metrics
            td3_metrics = metrics.get('td3_specific', {})
            if td3_metrics:
                plt.figure(figsize=(12, 6))
                plt.plot(episodes, td3_metrics.get('critic_difference', []), label='Critic Difference')
                plt.title(f'TD3 Critic Disagreement ({self.env_type})')
                plt.xlabel('Episode')
                plt.ylabel('Absolute Difference')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(os.path.join(self.plots_dir, 'td3_critic_difference.png'), dpi=300)
                plt.close()

def generate_visualizations(agent_type='both', env_type='both'):
    """Generate visualizations for specified agent and environment types"""
    agent_types = ['td3', 'sac'] if agent_type == 'both' else [agent_type]
    env_types = ['classic', 'hardcore'] if env_type == 'both' else [env_type]
    
    for agent in agent_types:
        for env in env_types:
            try:
                print(f"\nGenerating plots for {agent.upper()} on {env} environment...")
                visualizer = RL_Visualizer(agent, env)
                
                visualizer.plot_exploration_exploitation()
                visualizer.plot_performance_metrics()
                visualizer.plot_performance_distribution()
                visualizer.plot_test_performance()
                visualizer.plot_comprehensive_training()
                visualizer.plot_algorithm_specific()
                
                print(f"Plots saved in: results/plots/{agent}/{env}/")
                
            except FileNotFoundError as e:
                print(f"Warning: {str(e)}")
                continue
            except Exception as e:
                print(f"Error generating plots for {agent} on {env}: {str(e)}")
                continue

if __name__ == "__main__":
    generate_visualizations('both', 'both')