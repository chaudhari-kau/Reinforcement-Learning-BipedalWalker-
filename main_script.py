import gym
from gym.wrappers import Monitor
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from td3_agent import TD3Agent
from sac_agent import SACAgent
from train import train, test
from environment import MyWalker
import argparse
import os
from archs.ff_models import Actor, Critic
import time
from datetime import datetime, timedelta
import imageio
from utils import TimeEstimator

def print_menu(options):
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    choice = 0
    while choice < 1 or choice > len(options):
        try:
            choice = int(input(f"Select an option (1-{len(options)}): "))
        except ValueError:
            print("Please enter a valid number")
    return choice

def interactive_mode():
    start_time = time.time()
    print(f"Starting session at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main menu
    main_options = ["Train", "Evaluate", "Visualize"]
    main_choice = print_menu(main_options)
    
    # Environment selection
    env_options = ["BipedalWalker-v3", "BipedalWalkerHardcore-v3", "Both"]
    env_choice = print_menu(env_options)
    env_types = ['classic'] if env_choice == 1 else ['hardcore'] if env_choice == 2 else ['classic', 'hardcore']
    
    # Algorithm selection
    algo_options = ["TD3", "SAC", "Both"]
    algo_choice = print_menu(algo_options)
    agent_types = ['td3'] if algo_choice == 1 else ['sac'] if algo_choice == 2 else ['td3', 'sac']

    if main_choice == 1:  # Train
        for env_type in env_types:
            for agent_type in agent_types:
                operation_start = time.time()
                print(f"\nStarting Training for {env_type} with {agent_type}")
                
                # Setup environment and parameters based on type
                if env_type == 'classic':
                    env = gym.make('BipedalWalker-v3')
                    env = MyWalker(env, skip=2)
                    score_limit = 300.0
                    n_episodes = 4000
                    max_steps = 800
                else:
                    env = gym.make('BipedalWalkerHardcore-v3')
                    env = MyWalker(env, skip=2)
                    score_limit = 300.0
                    n_episodes = 8000
                    max_steps = 1000

                # Setup agent
                agent = setup_agent(agent_type, env)
                agent.train_mode()
                
                # Train
                scores, test_scores = train(env, agent, n_episodes=n_episodes, 
                                         model_type='ff', env_type=env_type, 
                                         score_limit=score_limit)
                
                # Save results
                save_results(scores, test_scores, env_type, agent_type)
                
                operation_time = time.time() - operation_start
                print(f"Training completed in {str(timedelta(seconds=int(operation_time)))}")
                
                # Close environment
                env.close()
            
    elif main_choice == 2:  # Evaluate
        for env_type in env_types:
            for agent_type in agent_types:
                operation_start = time.time()
                print(f"\nStarting Evaluation for {env_type} with {agent_type}")
                
                # Setup environment based on type
                if env_type == 'classic':
                    env = gym.make('BipedalWalker-v3')
                    env = MyWalker(env, skip=2)
                    max_steps = 800
                else:
                    env = gym.make('BipedalWalkerHardcore-v3')
                    env = MyWalker(env, skip=2)
                    max_steps = 1000

                # Setup agent
                agent = setup_agent(agent_type, env)
                
                # Load checkpoint
                checkpoint = input(f"Enter checkpoint number for {agent_type} on {env_type} (e.g., 6000 for ep6000): ")
                agent.load_ckpt('ff', env_type, f'ep{checkpoint}')
                agent.eval_mode()

                # Evaluation options
                eval_options = ["Single episode", "Multiple episodes (100)", "Record video"]
                eval_choice = print_menu(eval_options)

                if eval_choice == 1:
                    scores = test(env, agent, max_t_step=max_steps)
                elif eval_choice == 2:
                    scores = test(env, agent, render=False, n_times=100, max_t_step=max_steps)
                else:  # Record video
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gif_filename = f"{env_type}_{agent_type}_{checkpoint}_{timestamp}.gif"
                    record_episode(env, agent, env_type, agent_type, gif_filename, max_steps=max_steps)

                operation_time = time.time() - operation_start
                print(f"Evaluation completed in {str(timedelta(seconds=int(operation_time)))}")
                
                # Close environment
                env.close()

    elif main_choice == 3:  # Visualize
        operation_start = time.time()
        print(f"\nStarting visualization...")
                
        # Generate comparison plots between algorithms
        from comparisons import generate_comparison_plots
        # First generate comparison plots for all selected environments
        for env in env_types:
            print(f"\nGenerating comparison plots for {env}...")
            generate_comparison_plots(env)
                
        # Now generate individual algorithm visualizations
        from visualizer import RL_Visualizer
        for env in env_types:
            for agent in agent_types:
                try:
                    print(f"\nGenerating visualizations for {agent.upper()} on {env}...")
                    visualizer = RL_Visualizer(agent, env)
                    visualizer.plot_performance_metrics()
                    visualizer.plot_performance_distribution()
                    visualizer.plot_comprehensive_training()
                except Exception as e:
                    print(f"Error generating visualizations for {agent} on {env}: {str(e)}")
                
        operation_time = time.time() - operation_start
        print(f"\nVisualization completed in {str(timedelta(seconds=int(operation_time)))}")
    
    total_time = time.time() - start_time
    print(f"\nTotal session time: {str(timedelta(seconds=int(total_time)))}")

def record_episode(env, agent, env_type, agent_type, filename, max_steps=1000):
    """Record a single episode and save as GIF"""
    timer = TimeEstimator(max_steps, "Recording episode")
    
    # Create base directory structure
    base_dir = os.path.join('results', 'gifs')
    os.makedirs(base_dir, exist_ok=True)
    
    # Create environment type subdirectory
    env_type_dir = os.path.join(base_dir, env_type)
    os.makedirs(env_type_dir, exist_ok=True)
    
    # Create agent type subdirectory
    agent_dir = os.path.join(env_type_dir, agent_type)
    os.makedirs(agent_dir, exist_ok=True)
    
    frames = []
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        action = agent.get_action(state, explore=False)
        action = action.clip(min=env.action_space.low, max=env.action_space.high)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        timer.update(steps)
    
    env.close()
    
    # Save GIF in the proper directory with full path
    gif_path = os.path.join(agent_dir, filename)
    print(f"\nSaving video to {gif_path}...")
    
    # Ensure the frames are not empty before saving
    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"GIF saved successfully!")
    else:
        print("Warning: No frames were captured!")
        
    timer.finish()
    print(f"Episode recorded! Total reward: {total_reward:.2f}, Steps: {steps}")
    return total_reward, steps

def setup_agent(agent_type, env):
    if agent_type == 'td3':
        return TD3Agent(Actor, Critic, clip_low=-1, clip_high=+1, 
                       state_size=env.observation_space.shape[-1], 
                       action_size=env.action_space.shape[-1],
                       lr=4e-4)
    else:
        return SACAgent(Actor, Critic, clip_low=-1, clip_high=+1, 
                       state_size=env.observation_space.shape[-1], 
                       action_size=env.action_space.shape[-1],
                       lr=4e-4, alpha=0.01)

def save_results(scores, test_scores, env_type, agent_type):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(scores[0], scores[1], 'b.', alpha=0.5)
    ax.plot(scores[0], scores[2], 'b-', alpha=1.0, label='ff-'+agent_type)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title('Score History')
    ax.legend()
    fig.savefig(os.path.join("results", f'ff-{agent_type}.png'))
    plt.close()
    
    np.savetxt(os.path.join("results", f"train-{env_type}-ff-{agent_type}.txt"), scores, fmt="%.6e")
    np.savetxt(os.path.join("results", f"test-{env_type}-ff-{agent_type}.txt"), test_scores, fmt="%.6e")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-f", "--flag", type=str, choices=['train', 'test', 'test-record', 'test-exp', 'test-100'],
                        default='train', help="train or test?")
    parser.add_argument("-e", "--env", type=str, choices=['classic', 'hardcore'],
                        default='classic', help="environment type, classic or hardcore?")
    parser.add_argument("-r", "--rl_type", type=str, choices=['td3', 'sac'], 
                        default='sac', help='RL method')

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    # Command line mode
    if args.env == 'classic':
        env = gym.make('BipedalWalker-v3')
        env = MyWalker(env, skip=2)
        score_limit = 300.0
        n_episodes = 4000
        max_steps = 800
        env_type = "classic"
    else:
        env = gym.make('BipedalWalkerHardcore-v3')
        env = MyWalker(env, skip=2)
        score_limit = 300.0
        n_episodes = 8000
        max_steps = 1000
        env_type = "hardcore"

    agent = setup_agent(args.rl_type, env)
    agent.load_ckpt('ff', env_type, args.ckpt)

    print("Action dimension : ", env.action_space.shape)
    print("State  dimension : ", env.observation_space.shape)
    print("Action sample : ", env.action_space.sample())
    print("State sample  : \n ", env.reset())    

    if args.flag == 'train':
        agent.train_mode()   
        scores, test_scores = train(env, agent, n_episodes=n_episodes, model_type='ff', 
                                  env_type=env_type, score_limit=score_limit, 
                                  explore_episode=args.explore_episode)
        save_results(scores, test_scores, env_type, args.rl_type)

    elif args.flag == 'test' or args.flag == 'test-exp':
        agent.eval_mode()
        explore = (args.flag == 'test-exp')
        scores = test(env, agent, explore=explore, max_t_step=max_steps)
        env.close()

    elif args.flag == 'test-record':
        agent.eval_mode()
        record_dir = os.path.join('results', 'video', f'{env_type}_{args.rl_type}')
        os.makedirs(record_dir, exist_ok=True)
        env = Monitor(env, record_dir, force=True)
        scores = test(env, agent, max_t_step=max_steps)
        env.close()

    elif args.flag == 'test-100':
        agent.eval_mode()
        scores = test(env, agent, render=False, explore=False, n_times=100, max_t_step=max_steps)

    else:
        print('Wrong flag!')

if __name__ == "__main__":
    main()