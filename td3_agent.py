import torch
from torch import optim
import numpy as np
import os
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckNoise, DecayingOrnsteinUhlenbeckNoise, GaussianNoise
from itertools import chain

class TD3Agent():
    rl_type = 'td3'
    def __init__(self, Actor, Critic, clip_low, clip_high, state_size=24, action_size=4, update_freq=int(2),
            lr=3e-4, weight_decay=1e-2, gamma=0.99, tau=0.005, batch_size=256, buffer_size=int(1000000), device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.update_freq = update_freq
        self.learn_call = int(0)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        self.clip_low = torch.tensor(clip_low).to(self.device)
        self.clip_high = torch.tensor(clip_high).to(self.device)

        # Initialize actor networks
        self.train_actor = Actor().to(self.device)
        self.target_actor = Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor)
        self.actor_optim = torch.optim.AdamW(self.train_actor.parameters(), 
                                            lr=lr, 
                                            weight_decay=weight_decay, 
                                            amsgrad=True)
        print(f'Number of parameters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        # Initialize critic networks
        self.train_critic_1 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_1, self.target_critic_1)
        self.critic_1_optim = torch.optim.AdamW(self.train_critic_1.parameters(), 
                                               lr=lr, 
                                               weight_decay=weight_decay, 
                                               amsgrad=True)

        self.train_critic_2 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_2, self.target_critic_2)
        self.critic_2_optim = torch.optim.AdamW(self.train_critic_2.parameters(), 
                                               lr=lr, 
                                               weight_decay=weight_decay, 
                                               amsgrad=True)
        print(f'Number of parameters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}')

        # Initialize noise generators
        self.noise_generator = DecayingOrnsteinUhlenbeckNoise(
            mu=np.zeros(action_size),
            theta=0.15,
            sigma=0.3,
            dt=0.01,
            sigma_decay=0.9999
        )
        self.target_noise = GaussianNoise(
            mu=np.zeros(action_size),
            sigma=0.2,
            clip=0.4
        )
        
        # Initialize replay buffer and loss function
        self.memory = ReplayBuffer(action_size=action_size,
                                 buffer_size=buffer_size,
                                 batch_size=self.batch_size,
                                 device=self.device)
        
        self.huber_loss = torch.nn.SmoothL1Loss()

        # Initialize training metrics
        self.training_info = {
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
        
    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_one_step()

    def learn_one_step(self):
        if len(self.memory) > self.batch_size:
            exp = self.memory.sample()
            self.learn(exp)        
            
    def learn(self, exp):
        self.learn_call += 1
        states, actions, rewards, next_states, done = exp
        
        # Update critics
        with torch.no_grad():
            # Get next actions with noise for target policy smoothing
            next_actions = self.target_actor(next_states)
            noise = torch.clamp(
                torch.randn_like(next_actions) * 0.2,
                -0.5, 0.5
            )
            next_actions = torch.clamp(
                next_actions + noise,
                self.clip_low,
                self.clip_high
            )
            
            # Compute target Q values
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
        
        # Update Critic 1
        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = self.huber_loss(Q_expected_1, Q_targets)
        
        self.critic_1_optim.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        critic1_grad_norm = torch.nn.utils.clip_grad_norm_(self.train_critic_1.parameters(), 1.0)
        self.critic_1_optim.step()

        # Update Critic 2
        Q_expected_2 = self.train_critic_2(states, actions)   
        critic_2_loss = self.huber_loss(Q_expected_2, Q_targets)
        
        self.critic_2_optim.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        critic2_grad_norm = torch.nn.utils.clip_grad_norm_(self.train_critic_2.parameters(), 1.0)
        self.critic_2_optim.step()

        # Log critic metrics
        with torch.no_grad():
            self.training_info['critic1_losses'].append(critic_1_loss.item())
            self.training_info['critic2_losses'].append(critic_2_loss.item())
            self.training_info['q_values_mean'].append(Q_expected_1.mean().item())
            self.training_info['q_values_std'].append(Q_expected_1.std().item())
            self.training_info['target_q_values'].append(Q_targets.mean().item())
            self.training_info['gradient_norms']['critic1'].append(critic1_grad_norm.item())
            self.training_info['gradient_norms']['critic2'].append(critic2_grad_norm.item())
            self.training_info['td3_specific']['critic_difference'].append((Q_expected_1 - Q_expected_2).abs().mean().item())
            self.training_info['td3_specific']['target_noise_values'].append(self.target_noise().std())
            self.training_info['td3_specific']['action_noise_std'].append(self.noise_generator.sigma)
        
        # Delayed policy updates
        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0
            
            # Update actor
            actions_pred = self.train_actor(states)
            actor_loss = -self.train_critic_1(states, actions_pred).mean()
            
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.train_actor.parameters(), 1.0)
            self.actor_optim.step()

            # Log actor metrics
            with torch.no_grad():
                self.training_info['actor_losses'].append(actor_loss.item())
                self.training_info['policy_mean_values'].append(actions_pred.mean().item())
                self.training_info['gradient_norms']['actor'].append(actor_grad_norm.item())
                self.training_info['learning_rates'].append(self.actor_optim.param_groups[0]['lr'])
                self.training_info['td3_specific']['policy_delay_steps'].append(self.learn_call)
                self.training_info['td3_specific']['target_policy_smoothing'].append(1.0)
        
            # Update target networks
            self.soft_update(self.train_actor, self.target_actor)
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)
        
    @torch.no_grad()        
    def get_action(self, state, explore=False):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        action = self.train_actor(state).cpu().data.numpy()[0]

        if explore:
            noise = self.noise_generator()
            action = np.clip(action + noise, -1, 1)
        return action
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_ckpt(self, model_type, env_type, prefix='last'):
        save_dir = os.path.join("models", self.rl_type, env_type)
        os.makedirs(save_dir, exist_ok=True)
        
        actor_file = os.path.join(save_dir, "_".join([prefix, model_type, "actor.pth"]))
        critic_1_file = os.path.join(save_dir, "_".join([prefix, model_type, "critic_1.pth"]))
        critic_2_file = os.path.join(save_dir, "_".join([prefix, model_type, "critic_2.pth"]))
        
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)

    def load_ckpt(self, model_type, env_type, prefix='last'):
        actor_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "actor.pth"]))
        critic_1_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_1.pth"]))
        critic_2_file = os.path.join("models", self.rl_type, env_type, "_".join([prefix, model_type, "critic_2.pth"]))
        
        try:
            self.train_actor.load_state_dict(torch.load(actor_file, map_location=self.device))
            self.target_actor.load_state_dict(torch.load(actor_file, map_location=self.device))
        except:
            print("Actor checkpoint cannot be loaded.")
        try:
            self.train_critic_1.load_state_dict(torch.load(critic_1_file, map_location=self.device))
            self.train_critic_2.load_state_dict(torch.load(critic_2_file, map_location=self.device))
            self.target_critic_1.load_state_dict(torch.load(critic_1_file, map_location=self.device))
            self.target_critic_2.load_state_dict(torch.load(critic_2_file, map_location=self.device))
        except:
            print("Critic checkpoints cannot be loaded.")              

    def train_mode(self):
        self.train_actor.train()
        self.train_critic_1.train()
        self.train_critic_2.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic_1.eval()
        self.train_critic_2.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), 
                      self.train_critic_1.parameters(), 
                      self.train_critic_2.parameters()):
            p.requires_grad = False

    def step_end(self):
        self.noise_generator.step_end()

    def episode_end(self):
        self.noise_generator.episode_end()