import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
# import icu sepsis here
from icusepsisEnv import ICUSepsisEnv

# 716 possible states
# 25 admissible actions
STATE_DIM = 716               # One-hot encoded state representation
ACTION_DIM = 25               # 25 possible medical actions
TERMINAL_STATES = {713, 714}  # Terminal states for death and survival


# ICUSepsisEnv should throw an error for inadmissible action, ie, not within 0 to 24
# should also have reset() function to reset state to new episode
# function for step(action) which transitions to next state based on action
# ICUSepsisEnv can be from icu-sepsis package or from csv loaded into program
# Regardless, it needs a container class to simplify the program

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )
    
    def forward(self, x):
        return self.network(x)

class FederatedRLClient:
    def __init__(self, global_model, env):
        self.model = DQN()  # .eval() for inference
        self.model.load_state_dict(global_model.state_dict())
        self.env = env
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.87

    def _one_hot_encode(self, state):
        """Convert state integer to one-hot vector"""
        return np.eye(STATE_DIM)[state]

    def train(self, num_episodes):
        self.model.train()
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                state_oh = self._one_hot_encode(state)
                state_tensor = torch.FloatTensor(state_oh)
                
                # Epsilon-greedy action selection
                if np.random.rand() <= self.epsilon:
                    #action = self.env.action_space.sample()
                    action = np.random.randint(ACTION_DIM)
                else:
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        action = torch.argmax(q_values).item()
                
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition in replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Train on batches from replay buffer
                #if len(self.replay_buffer) >= self.batch_size:
                #    self._update_model()
            
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return self.model.state_dict()

    def _update_model(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to one-hot encoding
        states_oh = torch.FloatTensor(np.eye(STATE_DIM)[np.array(states)])
        next_states_oh = torch.FloatTensor(np.eye(STATE_DIM)[np.array(next_states)])
        
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # Calculate current Q values
        current_q = self.model(states_oh).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q values
        with torch.no_grad():
            next_q = self.model(next_states_oh).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to improve stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  
        self.optimizer.step()

def federated_average(weights_list):
    averaged_weights = {}
    for key in weights_list[0].keys():
        averaged_weights[key] = torch.mean(
            torch.stack([weights[key].float() for weights in weights_list]), dim=0
        )
    return averaged_weights

class FederatedRLServer:
    def __init__(self):
        self.global_model = DQN()
        self.num_clients = 2  # Number of hospitals/regions participating
        self.num_rounds = 50
        self.episodes_per_client = 1700

    def run_training(self):
        for round in range(self.num_rounds):
            client_weights = []
            
            # Simulate client training (TODO: add parallel command, or threading)
            # each client on different thread
            for _ in range(self.num_clients):
                # ICUSepsisEnv can be from icu-sepsis or a class created using csv loaded into program
                # TODO: add functions for reset() state and for step(action) which changes state for a given action
                client_env = ICUSepsisEnv()  
                client = FederatedRLClient(self.global_model, client_env)
                if len(client.replay_buffer) >= client.batch_size:
                    client._update_model()
                client_weights.append(client.train(self.episodes_per_client))
            
            # Aggregate and update global model
            averaged_weights = federated_average(client_weights)
            self.global_model.load_state_dict(averaged_weights)
            
            # Optional: Evaluate global model at steps
            test_reward = self.evaluate_model()
            print(f"Round {round+1} | Avg Test Reward: {test_reward:.2f}")
        
        return self.global_model

    def evaluate_model(self, num_episodes=200):
        # Evaluate the global model on a test environment
        self.global_model.eval()
        test_env = ICUSepsisEnv()
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = test_env.reset()
            done = False
            while not done:
                state_oh = np.eye(STATE_DIM)[state]
                with torch.no_grad():
                    q_values = self.global_model(torch.FloatTensor(state_oh))
                    action = torch.argmax(q_values).item()
                #action = np.random.randint(5)
                state, reward, done, _ = test_env.step(action)
                total_reward += reward
                
        return total_reward / num_episodes

if __name__ == "__main__":
    server = FederatedRLServer()
    trained_model = server.run_training()
    torch.save(trained_model.state_dict(), "icu_sepsis_fed_rl.pth")
