import pandas as pd
import numpy as np

class ICUSepsisEnv:
    TERMINAL_STATES = {713, 714}
    
    def __init__(self, transition_file='transitionFunction.csv', 
                 initial_file='initialStateDistribution.csv'):
        # Load transition probabilities
        self.transition_df = pd.read_csv(transition_file, header=None)
        self.transition_probs = self.transition_df.values
        
        # Load initial state distribution
        self.initial_dist = pd.read_csv(initial_file, header=None).values.flatten()
        # Validate dimensions
        self.num_states = self.initial_dist.shape[0]
        self.num_actions = self.transition_probs.shape[0] // self.num_states
        
        # Verify transition matrix dimensions
        assert self.transition_probs.shape == (self.num_states * self.num_actions, self.num_states), \
            "Transition matrix dimensions mismatch"
        # Initialize state tracking
        self.current_state = None
        
        # Define action space (discrete)
        #self.action_space = SimpleNamespace(
        #    n=self.num_actions,
        #    sample=lambda: np.random.randint(self.num_actions) 
        #    # could do randint(25) as well
        #)
        self.action_space = type('ActionSpace', (), {'n': self.num_actions})
        
        
    def reset(self):
        # Reset the environment to a random initial state
        self.current_state = np.random.choice(self.num_states, p=self.initial_dist)
        return self.current_state
    
    def step(self, action):
        # execute one step in the environment
        # Returns: (next_state, reward, done, info)
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action {action}. Must be 0-{self.num_actions-1}")
            
        # Get transition probabilities for current state and action
        row_idx = self.current_state * self.num_actions + action
        probs = self.transition_probs[row_idx]
        # Sample next state
        next_state = np.random.choice(self.num_states, p=probs)
        # Calculate reward
        reward = 1.0 if next_state == 714 else 0.0
        # Check termination
        done = next_state in self.TERMINAL_STATES
        # Update current state
        self.current_state = next_state if not done else None
       
        return next_state, reward, done, {}
    
    @property
    def observation_space(self):
        # return the observation space structure
        return type('ObservationSpace', (), {'n': self.num_states})
