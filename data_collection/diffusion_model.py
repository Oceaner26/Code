import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm # Import tqdm for progress bars in sampling

# Global constants for the diffusion schedule
# These constants define the underlying diffusion process used for both training and sampling.
T_DIFFUSION_STEPS = 1000 # Total number of theoretical diffusion steps (T)
BETA_START = 1e-4      # Starting beta value for the linear schedule
BETA_END = 0.02        # Ending beta value for the linear schedule

# --- Helper function to pre-compute diffusion schedule terms ---
# This function calculates the necessary coefficients for the diffusion process
# and will be used by both the training loop and the sampling method.
def get_diffusion_schedule(device):
    """
    Computes and returns the diffusion schedule terms on the specified device.
    """
    betas = torch.linspace(BETA_START, BETA_END, T_DIFFUSION_STEPS, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Pre-compute square roots for efficiency
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }

class DiffusionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, seq_len=10):
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.state_dim = state_dim

        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Joint state/action conditioning
        self.condition_embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal convolution network
        self.unet = nn.Sequential(
            nn.Conv1d(action_dim + 2 * hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.Mish(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish()
        )
        
        # Predict the denoising residual
        self.output_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, noisy_traj, conditions, timesteps):
        # Input shapes:
        # noisy_traj: [batch, seq_len, action_dim]
        # conditions: [batch, seq_len, state_dim]
        # timesteps:  [batch, 1]

        assert noisy_traj.shape[1] == self.seq_len
        assert conditions.shape[1] == self.seq_len

        # Match the device of the noisy action trajectory
        timesteps = timesteps.to(noisy_traj.device) 
        
        
        
        timesteps_float_on_device = timesteps.float().to(noisy_traj.device)
        
        
        t_emb = self.time_embed(timesteps_float_on_device) 
        
        # Embed the conditional inputs
        # Concatenate noisy actions and states along the feature dimension
        c_input = torch.cat([noisy_traj, conditions], dim=-1) # [batch, seq_len, action_dim + state_dim]
        c_emb = self.condition_embed(c_input) # [batch, seq_len, hidden_dim]
        
        # Broadcast timestep embedding across sequence positions
        t_emb_expanded = t_emb.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Concatenate all features for U-Net
        # Order: noisy_traj, condition_embedding, time_embedding
        x = torch.cat([noisy_traj, c_emb, t_emb_expanded], dim=-1) # [batch, seq_len, action_dim + hidden_dim + hidden_dim]
        
        # Permute for Conv1D: [batch, channels, seq_len]
        x = x.permute(0, 2, 1) 
        
        x = self.unet(x) # Output of unet: [batch, hidden_dim, seq_len]
        
        # Permute back: [batch, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)
        
        return self.output_layer(x) # [batch, seq_len, action_dim]

    def sample(self, condition, num_samples=5, steps=50):
        """
        Diffusion model sampling process - CORRECTED FOR SEQUENCES
        :param condition: Input state [1, state_dim] (current state)
        :param num_samples: Number of candidate trajectories to generate
        :param steps: Denoising steps
        :return: Generated action trajectories [num_samples, seq_len, action_dim]
        """
        from tqdm import tqdm # Import tqdm here if not imported globally
        
        # Ensure the condition tensor is on the model device
        device = next(self.parameters()).device 
        condition = condition.to(device)

        # Retrieve the diffusion schedule terms.
        betas = torch.linspace(BETA_START, BETA_END, T_DIFFUSION_STEPS, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Initialize random noise for the WHOLE sequence
        # x_t is the noisy trajectory at diffusion step t
        x_t = torch.randn(num_samples, self.seq_len, self.action_dim, device=device)

        # Expand the single current state 'condition' to match seq_len for model input
        condition_expanded_for_model = condition.expand(num_samples, -1) # [num_samples, state_dim]
        condition_expanded_for_model = condition_expanded_for_model.unsqueeze(1).expand(-1, self.seq_len, -1) # [num_samples, seq_len, state_dim]

        # Generate DDIM sampling timesteps.
        timesteps_to_sample = torch.linspace(T_DIFFUSION_STEPS - 1, 0, steps).long().to(device)

        for i, t_val in enumerate(tqdm(timesteps_to_sample, desc="Diffusion Sampling", leave=False, total=steps)):
            t_prev_val = timesteps_to_sample[i+1] if i < len(timesteps_to_sample) - 1 else torch.tensor(0, device=device)
            
            # Normalize the current timestep for the batch
            # Ensure this tensor is always on the correct device.
            timestep_norm = (t_val.float() / T_DIFFUSION_STEPS).unsqueeze(-1) # [1]
            timesteps_tensor = timestep_norm.expand(num_samples, -1) # [num_samples, 1]

            with torch.no_grad():
                pred_noise = self.forward(
                    x_t,                            # Current noisy sequence [num_samples, seq_len, action_dim]
                    condition_expanded_for_model,   # Conditioning sequence [num_samples, seq_len, state_dim]
                    timesteps_tensor                # Timesteps for batch [num_samples, 1]
                )

            # DDIM Denoising Step
            sqrt_alpha_bar_t = sqrt_alphas_cumprod[t_val]
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t_val]

            sqrt_alpha_bar_t_prev = sqrt_alphas_cumprod[t_prev_val]
            sqrt_one_minus_alpha_bar_t_prev = sqrt_one_minus_alphas_cumprod[t_prev_val]
            
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            
            x_t = sqrt_alpha_bar_t_prev * x_0_pred + sqrt_one_minus_alpha_bar_t_prev * pred_noise
            
        x_0 = torch.clamp(x_t, -1.0, 1.0)
        return x_0.detach()

def train_diffusion(dataset_path, device, seq_len=10):
    # Infer input dimensions from the dataset
    dataset = np.load(dataset_path, allow_pickle=True)
    sample_state = dataset['states'][0]
    sample_action = dataset['actions'][0]
    
    state_dim = sample_state.shape[-1]
    action_dim = sample_action.shape[-1]
    
    # Initialize the model
    model = DiffusionModel(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Preprocess training sequences
    states = torch.FloatTensor(dataset['states']).to(device)
    actions = torch.FloatTensor(dataset['actions']).to(device)
    
    # Pre-compute diffusion schedule terms once at the beginning of training
    diffusion_schedule = get_diffusion_schedule(device)
    sqrt_alphas_cumprod = diffusion_schedule['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_schedule['sqrt_one_minus_alphas_cumprod']
    
    # Train the denoising model
    for epoch in range(100000): # Consider a smaller number for initial tests
        idx = torch.randint(0, len(states) - seq_len, (32,)) # batch_size = 32
        batch_states_seq = torch.stack([states[i:i+seq_len] for i in idx])
        batch_actions_seq = torch.stack([actions[i:i+seq_len] for i in idx])

        # Sample timesteps uniformly for the batch
        sampled_timesteps_int = torch.randint(0, T_DIFFUSION_STEPS, (batch_actions_seq.size(0),), device=device)
        
        # Generate random noise (epsilon)
        noise = torch.randn_like(batch_actions_seq)
        
        # Retrieve the correct scaling factors for each sampled timestep from the schedule.
        alpha_bar_t_sqrt_batch = sqrt_alphas_cumprod[sampled_timesteps_int].unsqueeze(1).unsqueeze(1)
        one_minus_alpha_bar_t_sqrt_batch = sqrt_one_minus_alphas_cumprod[sampled_timesteps_int].unsqueeze(1).unsqueeze(1)
        
        # Add noise to batch_actions_seq (x_0) to create noisy_actions_seq (x_t)
        noisy_actions_seq = alpha_bar_t_sqrt_batch * batch_actions_seq + \
                            one_minus_alpha_bar_t_sqrt_batch * noise
        
        # Normalize the sampled integer timesteps to a [0, 1] range for the model's forward pass.
        sampled_timesteps_norm = (sampled_timesteps_int.float() / T_DIFFUSION_STEPS).unsqueeze(-1)
        
        # Forward pass: Model predicts the noise (epsilon) that was added.
        pred_noise = model(
            noisy_actions_seq,
            batch_states_seq,
            sampled_timesteps_norm # Pass normalized timesteps
        )
        
        # Calculate the Mean Squared Error loss between predicted noise and true noise.
        loss = F.mse_loss(pred_noise, noise)
        
        # Apply the gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    return model