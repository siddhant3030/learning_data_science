import gym
import numpy as np
import random

# Create the FrozenLake environment (non-slippery version for clarity)
env = gym.make('FrozenLake-v1', is_slippery=False)

# Print basic info about the environment
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

n_states = env.observation_space.n  # Expecting an integer number of states
n_actions = env.action_space.n
print("Number of states:", n_states)
print("Number of actions:", n_actions)

# Initialize the Q-table with zeros
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.8       # Learning rate
gamma = 0.95      # Discount factor
epsilon = 0.1     # Exploration rate
num_episodes = 2000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    # In some Gym versions, reset() might return a tuple (observation, info)
    if isinstance(state, tuple):
        state = state[0]
    print(f"\nEpisode {episode+1} start, initial state: {state} (type: {type(state)})")
    
    done = False
    step = 0
    while not done:
        print(f"  Episode {episode+1}, Step {step}: state = {state} (type: {type(state)})")
        
        # Ensure state is an integer
        if not isinstance(state, int):
            try:
                state = int(state)
                print(f"    Converted state to int: {state}")
            except Exception as e:
                print(f"    Could not convert state {state} to int: {e}")
        
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: random action
            print(f"    Chosen action (explore): {action}")
        else:
            action = np.argmax(Q[state])  # Exploit: best known action
            print(f"    Chosen action (exploit): {action}")
        
        # Take the action using the new API format
        result = env.step(action)
        print(f"    Raw result from env.step: {result}")
        
        # Handle both the 5-value (new) and 4-value (old) API cases
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            next_state, reward, done, info = result
        else:
            raise ValueError(f"Unexpected number of values returned from env.step: {len(result)}")
        
        print(f"    After action: next_state = {next_state}, reward = {reward}, done = {done}, info = {info}")

        # If next_state is a tuple, extract the observation
        if isinstance(next_state, tuple):
            next_state = next_state[0]
            print(f"    Extracted next_state from tuple: {next_state}")

        # Ensure next_state is an integer
        if not isinstance(next_state, int):
            try:
                next_state = int(next_state)
                print(f"    Converted next_state to int: {next_state}")
            except Exception as e:
                print(f"    Could not convert next_state {next_state} to int: {e}")

        # Q-learning update rule
        old_value = Q[state, action]
        Q[state, action] = old_value + alpha * (reward + gamma * np.max(Q[next_state]) - old_value)
        print(f"    Updated Q[{state}, {action}]: {old_value} -> {Q[state, action]}")
        
        state = next_state
        step += 1
        
        # Optional: Prevent runaway episodes during debugging
        if step > 50:
            print("    Breaking out of loop to prevent infinite steps")
            break

print("\nTraining complete. Final Q-table:")
print(Q)

# Testing the learned policy
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
print("\n--- Testing Phase ---")
print("Initial state:", state)
env.render()
done = False
step = 0
while not done:
    print(f"Test Step {step}: state = {state} (type: {type(state)})")
    if not isinstance(state, int):
        try:
            state = int(state)
            print(f"  Converted state to int: {state}")
        except Exception as e:
            print(f"  Could not convert state {state} to int: {e}")
    
    action = np.argmax(Q[state])
    print(f"Test Step {step}: chosen action = {action}")
    
    result = env.step(action)
    print(f"    Raw result from env.step in test: {result}")
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        done = terminated or truncated
    elif len(result) == 4:
        state, reward, done, info = result
    else:
        raise ValueError(f"Unexpected number of values returned from env.step: {len(result)}")
    
    if isinstance(state, tuple):
        state = state[0]
        print(f"  Extracted state from tuple: {state}")
    env.render()
    step += 1
    if step > 50:
        print("Breaking out of test loop to prevent infinite steps")
        break
