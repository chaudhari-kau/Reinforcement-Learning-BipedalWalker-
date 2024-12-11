import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime
from utils import TimeEstimator

def save_training_log(log_data, model_type, env_type, rl_type):
    """Save training logs to a JSON file"""
    log_dir = os.path.join("logs", rl_type)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{env_type}_{timestamp}.json"
    
    with open(os.path.join(log_dir, filename), 'w') as f:
        json.dump(log_data, f, indent=4)

def log_td3_metrics(training_log, agent):
    """Log TD3 specific training metrics"""
    with torch.no_grad():
        metrics = training_log['td3_metrics']
        
        # Get latest values from agent's current training step
        actor_loss = agent.actor_loss if hasattr(agent, 'actor_loss') else 0.0
        critic1_loss = agent.critic1_loss if hasattr(agent, 'critic1_loss') else 0.0
        critic2_loss = agent.critic2_loss if hasattr(agent, 'critic2_loss') else 0.0
        q_values = agent.q_values.float() if hasattr(agent, 'q_values') else torch.tensor([0.0])
        target_q = agent.target_q if hasattr(agent, 'target_q') else 0.0
        
        # Log basic losses and values
        metrics['actor_losses'].append(float(actor_loss))
        metrics['critic1_losses'].append(float(critic1_loss))
        metrics['critic2_losses'].append(float(critic2_loss))
        metrics['q_values_mean'].append(q_values.float().mean().item())
        metrics['q_values_std'].append(q_values.float().std().item() if len(q_values) > 1 else 0.0)
        metrics['target_q_values'].append(float(target_q))
        
        # Log learning rates
        metrics['learning_rates'].append(float(agent.actor_optim.param_groups[0]['lr']))
        
        # Log gradient norms - ensure tensors are float
        actor_norm = torch.nn.utils.clip_grad_norm_(agent.train_actor.parameters(), float('inf'))
        critic1_norm = torch.nn.utils.clip_grad_norm_(agent.train_critic_1.parameters(), float('inf'))
        critic2_norm = torch.nn.utils.clip_grad_norm_(agent.train_critic_2.parameters(), float('inf'))
        
        metrics['gradient_norms']['actor'].append(float(actor_norm))
        metrics['gradient_norms']['critic1'].append(float(critic1_norm))
        metrics['gradient_norms']['critic2'].append(float(critic2_norm))
        
        # TD3 specific metrics
        noise_std = agent.target_noise().std()
        metrics['td3_specific']['target_noise_values'].append(float(noise_std))
        metrics['td3_specific']['policy_delay_steps'].append(int(agent.learn_call))
        metrics['td3_specific']['action_noise_std'].append(float(agent.noise_generator.sigma))
        metrics['td3_specific']['target_policy_smoothing'].append(float(agent.learn_call % agent.update_freq == 0))

def log_sac_metrics(training_log, agent):
    """Log SAC specific training metrics"""
    with torch.no_grad():
        metrics = training_log['sac_metrics']
        
        # Get latest values from agent's current training step
        actor_loss = agent.actor_loss if hasattr(agent, 'actor_loss') else 0.0
        critic1_loss = agent.critic1_loss if hasattr(agent, 'critic1_loss') else 0.0
        critic2_loss = agent.critic2_loss if hasattr(agent, 'critic2_loss') else 0.0
        q_values = agent.q_values.float() if hasattr(agent, 'q_values') else torch.tensor([0.0])
        target_q = agent.target_q if hasattr(agent, 'target_q') else 0.0
        entropy = agent.entropy.float() if hasattr(agent, 'entropy') else torch.tensor([0.0])
        
        # Log basic losses and values
        metrics['actor_losses'].append(float(actor_loss))
        metrics['critic1_losses'].append(float(critic1_loss))
        metrics['critic2_losses'].append(float(critic2_loss))
        metrics['q_values_mean'].append(q_values.float().mean().item())
        metrics['q_values_std'].append(q_values.float().std().item() if len(q_values) > 1 else 0.0)
        metrics['target_q_values'].append(float(target_q))
        
        # Log entropy related metrics
        metrics['entropy_values'].append(entropy.float().mean().item())
        metrics['entropy_losses'].append(float(-agent.alpha * entropy.float().mean().item()))
        metrics['entropy_coefficients'].append(float(agent.alpha))
        
        # Log policy statistics
        policy_mean = agent.policy_mean if hasattr(agent, 'policy_mean') else 0.0
        policy_std = agent.policy_std if hasattr(agent, 'policy_std') else 0.0
        metrics['policy_mean_values'].append(float(policy_mean))
        metrics['policy_std_values'].append(float(policy_std))
        
        # Log learning rates
        metrics['learning_rates'].append(float(agent.actor_optim.param_groups[0]['lr']))
        
        # Log gradient norms
        actor_norm = torch.nn.utils.clip_grad_norm_(agent.train_actor.parameters(), float('inf'))
        critic1_norm = torch.nn.utils.clip_grad_norm_(agent.train_critic_1.parameters(), float('inf'))
        critic2_norm = torch.nn.utils.clip_grad_norm_(agent.train_critic_2.parameters(), float('inf'))
        
        metrics['gradient_norms']['actor'].append(float(actor_norm))
        metrics['gradient_norms']['critic1'].append(float(critic1_norm))
        metrics['gradient_norms']['critic2'].append(float(critic2_norm))
        
        # Log detailed entropy statistics
        if len(entropy) > 1:
            metrics['policy_entropy_stats']['min'].append(float(entropy.min()))
            metrics['policy_entropy_stats']['max'].append(float(entropy.max()))
            metrics['policy_entropy_stats']['mean'].append(float(entropy.mean()))
            metrics['policy_entropy_stats']['std'].append(float(entropy.std()))
        else:
            metrics['policy_entropy_stats']['min'].append(0.0)
            metrics['policy_entropy_stats']['max'].append(0.0)
            metrics['policy_entropy_stats']['mean'].append(0.0)
            metrics['policy_entropy_stats']['std'].append(0.0)

def train(env, agent, n_episodes=8000, model_type='unk', env_type='unk', 
          score_limit=300.0, explore_episode=50, test_f=200, save_freq=50):
    
    # Create necessary directories first
    os.makedirs(os.path.join("models", agent.rl_type, env_type), exist_ok=True)
    os.makedirs(os.path.join("results", "plots", agent.rl_type), exist_ok=True)
    
    timer = TimeEstimator(n_episodes, f"Training {env_type} environment with {agent.rl_type}")
    max_steps = 2000 if 'Hardcore' in env.spec.id else 1600
    
    # Initialize training variables
    scores_deque = deque(maxlen=100)
    scores = []
    test_scores = []
    max_score = -np.Inf
    
    # Initialize training log with basic structure
    training_log = {
        'episodes': [],
        'scores': [],
        'avg_scores': [],
        'test_scores': [],
        'episode_lengths': [],
        'max_scores': [],
        'metadata': {
            'model_type': model_type,
            'env_type': env_type,
            'rl_type': agent.rl_type,
            'total_episodes': n_episodes,
            'score_limit': score_limit,
            'explore_episodes': explore_episode,
            'max_steps': max_steps
        }
    }

    # Add algorithm-specific metrics
    if agent.rl_type == 'td3':
        training_log.update({
            'td3_metrics': {
                'actor_losses': [],
                'critic1_losses': [],
                'critic2_losses': [],
                'policy_mean_values': [],
                'q_values_mean': [],
                'q_values_std': [],
                'target_q_values': [],
                'learning_rates': [],
                'gradient_norms': {
                    'actor': [],
                    'critic1': [],
                    'critic2': []
                },
                'td3_specific': {
                    'target_noise_values': [],
                    'policy_delay_steps': [],
                    'critic_difference': [],
                    'action_noise_std': [],
                    'target_policy_smoothing': []
                }
            }
        })
    elif agent.rl_type == 'sac':
        training_log.update({
            'sac_metrics': {
                'actor_losses': [],
                'critic1_losses': [],
                'critic2_losses': [],
                'entropy_values': [],
                'entropy_losses': [],
                'entropy_coefficients': [],
                'policy_mean_values': [],
                'policy_std_values': [],
                'q_values_mean': [],
                'q_values_std': [],
                'target_q_values': [],
                'learning_rates': [],
                'gradient_norms': {
                    'actor': [],
                    'critic1': [],
                    'critic2': []
                },
                'policy_entropy_stats': {
                    'min': [],
                    'max': [],
                    'mean': [],
                    'std': []
                }
            }
        })

    # Add reward components and episode stats
    training_log.update({
        'reward_components': {
            'forward_rewards': [],
            'energy_penalties': [],
            'fall_penalties': []
        },
        'episode_stats': {
            'falls': [],
            'successful_completions': [],
            'average_speed': []
        }
    })
    
    try:
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            done = False
            agent.train_mode()
            t = int(0)
            episode_falls = 0
            forward_reward = 0
            energy_penalty = 0
            
            while not done and t < max_steps:    
                t += int(1)
                action = agent.get_action(state, explore=True)
                action = action.clip(min=env.action_space.low, max=env.action_space.high)
                next_state, reward, done, info = env.step(action)
                
                # Track reward components
                if info.get("dead", False):
                    episode_falls += 1
                
                agent.memory.add(state, action, reward, next_state, info["dead"])
                state = next_state
                score += reward
                agent.step_end()

            if i_episode > explore_episode:
                agent.episode_end()
                for i in range(t):
                    agent.learn_one_step()

            scores_deque.append(score)
            avg_score_100 = np.mean(scores_deque)
            scores.append((i_episode, score, avg_score_100))
            
            # Update basic metrics
            training_log['episodes'].append(i_episode)
            training_log['scores'].append(float(score))
            training_log['avg_scores'].append(float(avg_score_100))
            training_log['episode_lengths'].append(t)
            training_log['max_scores'].append(float(max(scores_deque)))
            
            # Update reward components and episode stats
            training_log['reward_components']['forward_rewards'].append(float(forward_reward))
            training_log['reward_components']['energy_penalties'].append(float(energy_penalty))
            training_log['episode_stats']['falls'].append(episode_falls)
            training_log['episode_stats']['successful_completions'].append(1 if score > 0 and not done else 0)
            
            # Update algorithm-specific metrics
            if agent.rl_type == 'td3':
                log_td3_metrics(training_log, agent)
            elif agent.rl_type == 'sac':
                log_sac_metrics(training_log, agent)
            
            # Update progress
            timer.update(i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tSteps: {}'.format(
                i_episode, avg_score_100, score, t), end="")

            # Save checkpoint every save_freq episodes
            if i_episode % save_freq == 0:
                checkpoint_prefix = f'ep{i_episode}'
                agent.save_ckpt(model_type, env_type, checkpoint_prefix)

            if i_episode % test_f == 0 or avg_score_100 > score_limit:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                agent.eval_mode()
                test_score = test(env, agent, render=False, n_times=20, max_t_step=max_steps)
                test_scores.append((i_episode, test_score))
                training_log['test_scores'].append((int(i_episode), float(test_score)))
                
                agent.save_ckpt(model_type, env_type, f'ep{int(i_episode)}')
                
                if avg_score_100 > score_limit:
                    print(f"\nEnvironment solved in {i_episode} episodes!")
                    break
                agent.train_mode()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        timer.finish()
        # Save final checkpoint and logs
        agent.save_ckpt(model_type, env_type, 'final')
        save_training_log(training_log, model_type, env_type, agent.rl_type)
    
    return np.array(scores).transpose(), np.array(test_scores).transpose()

def test(env, agent, render=True, max_t_step=1000, explore=False, n_times=1):
    timer = TimeEstimator(n_times, "Testing episodes")
    sum_scores = 0
    episode_data = []
    
    for i in range(n_times):
        state = env.reset()
        score = 0
        done = False
        t = int(0)
        episode_info = {
            'steps': 0,
            'falls': 0,
            'completion': False
        }
        
        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=explore)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            next_state, reward, done, info = env.step(action)
            
            if info.get("dead", False):
                episode_info['falls'] += 1
            
            state = next_state
            score += reward
            if render:
                env.render()
        
        episode_info['steps'] = t
        episode_info['completion'] = not info.get("dead", False)
        episode_data.append(episode_info)
        
        sum_scores += score
        timer.update(i + 1)
    
    mean_score = sum_scores/n_times
    timer.finish()
    
    # Print detailed test results
    print(f'\nTest Results over {n_times} episodes:')
    print(f'Mean Score: {mean_score:.2f}')
    print(f'Average Steps: {np.mean([ep["steps"] for ep in episode_data]):.1f}')
    print(f'Success Rate: {np.mean([ep["completion"] for ep in episode_data])*100:.1f}%')
    print(f'Average Falls per Episode: {np.mean([ep["falls"] for ep in episode_data]):.2f}')
    
    return mean_score