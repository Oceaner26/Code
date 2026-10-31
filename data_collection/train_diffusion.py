import os, sys
import pickle
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)
import math
from env import Env
import numpy as np
import argparse
import torch
import copy
from record import Record_object
from diffusion_model import train_diffusion, DiffusionModel
from td3 import TD3
import time
import pickle
import gc
from tqdm import tqdm
from env import Env
import jonswap_wave_current as wave
import warnings
warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
# ------ training paras ------
# Diffusion sampling options
parser.add_argument('--use_diffusion', type=int, default=1, help='Use diffusion model for action generation')
parser.add_argument('--diffusion_steps', type=int, default=20, help='Prediction steps for diffusion model')
parser.add_argument('--num_candidates', type=int, default=15, help='Number of candidate trajectories')

parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0008)
parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--replay_capa', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--policy_freq', type=int, default=2)
parser.add_argument('--episode_num', type=int, default=2002)
parser.add_argument('--episode_length', type=int, default=1000, help='the length of an episode (sec)')
parser.add_argument('--save_model_freq', type=int, default=25)
# ------ env paras ------
parser.add_argument('--R_dc', type=float, default=18., metavar='R_DC',help='the radius of data collection')
parser.add_argument('--border_x', type=float, default=200.,help='Area x size')
parser.add_argument('--border_y', type=float, default=200.,help='Area y size')
parser.add_argument('--border_z', type=float, default=200.,help='Area z size')
parser.add_argument('--wave_resolution', type=float, default=1.5,help='Min grid')
parser.add_argument('--n_s', type=int, default=60, help='The number of SNs')
parser.add_argument('--N_AUV', type=int, default=2, help='The number of AUVs')
parser.add_argument('--Q', type=float, default=2, help='Capacity of SNs (Mbits)')
parser.add_argument('--alpha', type=float, default=0.05, help='SNs choosing distance priority')
parser.add_argument('--step_size', type=float, default=1.0, help='Time per RL control step (s)')
parser.add_argument('--sim_step_size', type=float, default=0.05, help='Time per simulation step (s). Make sure this value can divide into `--step_size`.')
parser.add_argument('--control_mode', type=str, default="Ssurface_nref_nv")
parser.add_argument('--reward_ratio', type=float, default=1e4)
parser.add_argument('--k_yaw', type=str, default='[5, 1, 0.01]',help='Input list of length 2 (S-Surface) or 3 (PID)')
parser.add_argument('--k_depth', type=str, default='[1, 1, 0.01]',help='Input list of length 2 (S-Surface) or 3 (PID)')
parser.add_argument('--wave',action='store_true',help='Simualte JONSWAP wave and current.')
parser.add_argument('--usbl',action='store_true',help='Simualte USV-AUV collaboration and USBL positioning.')


args = parser.parse_args()
print(f"[MainScript_DEBUG] args.diffusion_steps: {args.diffusion_steps}")
print(f"[MainScript_DEBUG] args.num_candidates: {args.num_candidates}")
all_control_methods = {'SMC':'PVS original code implementation',
                       'Ssurface':'S surface controller (ref model)',
                       'Ssurface_nref':'S surface controller (w/o ref model)',
                       'Ssurface_nref_nv':'S surface controller (w/o ref model, not using velocity as Δe)',
                       'PID_nref':'PID controller (w/o ref model)',
                       'PID_nref_nv':'PID controller (w/o ref model, not using velocity as Δe)'}
if args.control_mode not in list(all_control_methods.keys()):
    raise NotImplementedError(f'All available controller are listed as below: \n {all_control_methods.__repr__()}')

try:
    k_yaw = eval(args.k_yaw); k_depth = eval(args.k_depth)
    if type(k_yaw) != list or type(k_depth) != list:
        raise TypeError
except:
    raise TypeError("Input `k_yaw` and `k_depth` type invalid.")

if len(k_yaw) < 3: k_yaw += [0] * (3 - len(k_yaw))
if len(k_depth) < 3: k_yaw += [0] * (3 - len(k_depth))


SAVE_PATH = ROOT_PATH + "/models_ddpg/"
REC_PATH = ROOT_PATH + "/record_train/"
LOG_PATH = ROOT_PATH + "/log/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(REC_PATH):
    os.makedirs(REC_PATH)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# Load the pretrained diffusion model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
diffusion_model = None
if args.use_diffusion:
    env = Env(args)
    diffusion_model = DiffusionModel(
        state_dim= env.state_dim, 
        action_dim=3,
        hidden_dim=256
    )
    
    
    checkpoint = torch.load(
        'diffusion_model.pth',
        weights_only=True,  
        map_location=device  
    )

    diffusion_model.load_state_dict(checkpoint)
    diffusion_model.to(device)
    diffusion_model.eval()

def train():
    init_formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    dataset_dir = "datasets"
    os.makedirs(dataset_dir, exist_ok=True)

    # Exploration noise schedule
    initial_noise_std = 0.25
    final_noise_std = 0.01
    decay_episodes = 1500
    action_dim = 3 

    # Specify episodes to archive for inspection
    EPISODE_TO_RECORD_FOR_ANALYSIS1 = 70
    EPISODE_TO_RECORD_FOR_ANALYSIS2 = 100
    # =======================================================================

    for ep in range(args.episode_num):
        state_c = env.reset()
        state = copy.deepcopy(state_c)
        record_object = Record_object(env, N_AUV)
        hovers = np.zeros(N_AUV)
        mode = np.zeros(N_AUV)
        ht = np.zeros(N_AUV)
        rand_phase = np.random.rand(wave.n_freq, wave.n_dir) * 2 * np.pi
        
        episode_data = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []
        }

        current_noise_std = max(
            final_noise_std,
            initial_noise_std - (initial_noise_std - final_noise_std) * (ep / decay_episodes)
        )
        
        for _ in tqdm(range(int(args.episode_length / args.step_size))):
            record_hover = np.zeros(N_AUV)
            td_loss = np.zeros(N_AUV)
            a_loss = np.zeros(N_AUV)
            
            act = []
            for i in range(N_AUV):
                if args.use_diffusion:
                    state_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        candidate_trajs = diffusion_model.sample(
                            condition=state_tensor,
                            num_samples=args.num_candidates,
                            steps=args.diffusion_steps
                        )

                    best_q = -float('inf')
                    best_first_action = None
                    for k in range(candidate_trajs.shape[0]):
                        first_action_of_traj = candidate_trajs[k, 0, :].unsqueeze(0).to(device)
                        with torch.no_grad():
                            q_val1, q_val2 = agents[i].critic(state_tensor, first_action_of_traj)
                            q_value = torch.min(q_val1, q_val2).item()
                        
                        if q_value > best_q:
                            best_q = q_value
                            best_first_action = first_action_of_traj.cpu().numpy().flatten()

                    noise = np.random.normal(0, current_noise_std, size=action_dim)
                    action_with_exploration = (best_first_action + noise).clip(-1.0, 1.0)
                    act.append(action_with_exploration)
                else:
                    act.append(agents[i].select_action(state[i]))
            
            env.posit_change(act, hovers, k_yaw=k_yaw, k_depth=k_depth, 
                            rand_phase=rand_phase, control=args.control_mode)
            state_, rewards, Done, data_rate, ec, cs = env.step(hovers)
            
            for i in range(N_AUV):
                episode_data['states'].append(state[i].copy())
                episode_data['actions'].append(np.array(act[i]).copy())
                episode_data['rewards'].append(float(rewards[i]))
                episode_data['next_states'].append(state_[i].copy())
                episode_data['dones'].append(bool(Done[i]))
            
            for i in range(N_AUV):
                if mode[i] == 0:
                    agents[i].store_transition(state[i], act[i], rewards[i], state_[i], False)
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i]:
                        record_object.idu[i] += 1
                        ht[i] = args.Q * env.updata[i] / data_rate[i]
                        mode[i] += math.ceil(ht[i])
                        hovers[i] = True
                        record_hover[i] = True
                else:
                    mode[i] -= args.step_size
                    record_object.Ht[i] += args.step_size
                    if mode[i] <= 0:
                        hovers[i] = False
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=args.alpha)
                
                if len(agents[i].replay_buffer) > 20 * args.batch_size:
                    a_loss[i], td_loss[i] = agents[i].train()
            
            
            record_object.update_metric(env, data_rate, record_hover, td_loss, a_loss)

        
        if ep % 100 == 0:
            print(f"\nEP {ep} | Current Exploration Noise STD: {current_noise_std:.4f}")

        # Save the selected episode records
        
        if ep == EPISODE_TO_RECORD_FOR_ANALYSIS1 or ep == EPISODE_TO_RECORD_FOR_ANALYSIS2:
            # Name each saved record using its episode number
            record_filename = f'record_object_ep{ep}.pkl'
            
            # Create the output directory if necessary
            os.makedirs(REC_PATH, exist_ok=True)
            
            # Serialize the episode record
            with open(os.path.join(REC_PATH, record_filename), 'wb') as f:
                pickle.dump(record_object, f)
            
            print(f"\n--- [RECORDING SAVED] Full record object for episode {ep} has been saved to: {os.path.join(REC_PATH, record_filename)} ---")
        # =====================================================================================

        
        if (ep % 2 == 0 and ep != 0) or (0 < ep < 5):
             formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
             with open(REC_PATH + f'/TRAIN_{ep}_TIME{formatted_time}.pkl', 'wb') as f:
                 pickle.dump(record_object, f)
        
        
        output_str = f'\rEP {ep} | ' + record_object.print_metric()
        print(output_str)
        with open(LOG_PATH + f'train_{init_formatted_time}.txt', 'a') as f:
            f.write(output_str + '\n')
        del record_object
        gc.collect()

    final_data_path = os.path.join(dataset_dir, "final_dataset")
    os.makedirs(final_data_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(final_data_path, "final_data.npz"),
        states=np.array(episode_data['states']),
        actions=np.array(episode_data['actions']),
        rewards=np.array(episode_data['rewards']),
        next_states=np.array(episode_data['next_states']),
        dones=np.array(episode_data['dones'])
    )


if __name__ == "__main__":
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 3 
    # Initialize the diffusion model
    diffusion_model = DiffusionModel(
        state_dim=state_dim,  
        action_dim=3,
        hidden_dim=256
    )
    agents = [TD3(env.state_dim, action_dim) for _ in range(N_AUV)]

    

    
    if not args.use_diffusion:
       train()
    else:
       # Use the same environment state dimensions at inference
       print("Using diffusion model for action generation")
       train()

    # Train the diffusion model using the provided dataset
    if args.is_train:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("train_diffusion")
        diffusion_model = train_diffusion(
            dataset_path='merged_dataset\merged_ep300-818_0616.npz',
            device=device,
            seq_len=args.diffusion_steps
        )
        torch.save(diffusion_model.state_dict(), 'diffusion_model.pth')

    