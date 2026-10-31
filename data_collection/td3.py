import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from diffusion_model import DiffusionModel  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DDPG
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # ---- Q1 ----
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        # ---- Q2 ----
        self.fc3 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        
        
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        # Q2
        x = F.relu(self.fc3(cat))
        x = F.relu(self.fc4(x))
        Q2 = self.fc_out2(x)
        return Q1, Q2

    # only return Q1 value(https://github.com/sfujim/TD3)
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        return Q1


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
    def save_dataset(self, filename):
        dir_path = os.path.dirname(filename) or '.'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert replay transitions to NumPy arrays
        data_dict = {
            'states': np.array([t[0] for t in self.buffer]),
            'actions': np.array([t[1] for t in self.buffer]),
            'rewards': np.array([t[2] for t in self.buffer]),
            'next_states': np.array([t[3] for t in self.buffer]),
            'dones': np.array([t[4] for t in self.buffer])
        }
        np.save(filename, data_dict)


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        discount=0.97,
        tau=0.001,
        lr=1e-3,
        batch_size=64,
        diffusion_model_path=None,
        use_diffusion=True
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.tau = tau
        self.replay_buffer = ReplayBuffer(20000)
        self.policy_freq = 2
        self.total_it = 0
        self.batch_size = batch_size

        # Optional diffusion proposal support
        self.use_diffusion = use_diffusion
        if diffusion_model_path:
            self.diffusion_model = self._load_diffusion_model(
                diffusion_model_path, 
                state_dim,
                action_dim,
                hidden_dim
            )
        else:
            self.diffusion_model = None

    def _load_diffusion_model(self, model_path, state_dim, action_dim, hidden_dim):
        """Load a pretrained diffusion model."""
        model = DiffusionModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Keep the loaded proposal model in evaluation mode
        return model
        
    def select_action(self, state, exploration=False):
        """Select an action from diffusion proposals or the original TD3 actor.

        :param exploration: If True, use the original TD3 actor branch.
        """
        if self.diffusion_model and self.use_diffusion and not exploration:
            return self._select_action_with_diffusion(state)
        else:
            return self._select_action_original(state)
        
    def _select_action_original(self, state):
        """Select an action directly from the TD3 actor."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu().numpy().flatten()
    
    def _select_action_with_diffusion(self, state):
        """Select an action using diffusion proposals."""
        # Generate candidate actions
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            
            # Sample five candidate actions
            candidate_actions = self.diffusion_model.sample(
                condition=state_tensor,
                num_samples=5
            )  # [5, action_dim]

            # Score candidates with the critic
            state_repeated = state_tensor.repeat(5, 1)
            q_values = self.critic.Q1(state_repeated, candidate_actions)
            
            # Execute the highest-scored candidate
            best_idx = torch.argmax(q_values)
            return candidate_actions[best_idx].cpu().numpy()
        
    def load_diffusion(self, model_path, device):
        """Load the diffusion model after initialization."""
        self.diffusion_model = DiffusionModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        ).to(device)
        self.diffusion_model.load_state_dict(torch.load(model_path))
        self.use_diffusion = True

    def store_transition(self, state, action, reward, next_state, done=False):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size=self.batch_size
        )
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device).unsqueeze(1)
        # Critic loss
        with torch.no_grad():
            policy_noise = 0.15
            noise_clip = 0.35
            # ------------------------------------

            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            # =================================================================

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self.total_it += 1
        # loss
        try:
            return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
        except:
            return 0, critic_loss.detach().cpu().numpy()

    # save model
    def save(self, filename, ep, idx):
        ep = "_" + str(ep)
        idx = "_" + str(idx)
        torch.save(
            self.critic.state_dict(), filename + "TD3" + idx + ep + "_critic.pth"
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "TD3" + idx + ep + "_critic_optimizer.pth",
        )
        torch.save(self.actor.state_dict(), filename + "TD3" + idx + ep + "_actor.pth")
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "TD3" + idx + ep + "_actor_optimizer.pth",
        )

    # load model
    def load(self, filename, ep, idx):
        ep = "_" + str(ep)
        idx = "_" + str(idx)
        self.critic.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_critic.pth", weights_only=True)
        )
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_critic_optimizer.pth", weights_only=True)
        )
        self.actor.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_actor.pth", weights_only=True)
        )
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "TD3" + idx + ep + "_actor_optimizer.pth", weights_only=True)
        )
        # target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
