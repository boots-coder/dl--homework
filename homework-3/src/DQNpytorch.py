import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

# ReplayBuffer class for storing and sampling experience transitions
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)  # Initialize buffer with a max size

    def put(self, transition):
        self.buffer.append(transition)  # Add a transition to the buffer

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # Randomly sample n transitions from the buffer
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # Convert lists to torch tensors
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)  # Return the current size of the buffer

# Neural network for approximating the Q-value function
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input layer (4 features) to hidden layer (128 units)
        self.fc2 = nn.Linear(128, 128)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(128, 2)  # Hidden layer to output layer (2 actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)  # Output layer (Q-values for each action)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)  # Forward pass through the network
        coin = random.random()  # Random value for epsilon-greedy action selection
        if coin < epsilon:
            return random.randint(0, 1)  # Random action
        else:
            return out.argmax().item()  # Greedy action

# Function to train the Q-network
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)  # Sample a batch of transitions from memory

        q_out = q(s)  # Q-values for the current states
        q_a = q_out.gather(1, a)  # Q-values for the taken actions
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # Max Q-values for the next states
        target = r + gamma * max_q_prime * done_mask  # Compute the target Q-value
        loss = F.smooth_l1_loss(q_a, target)  # Compute the Huber loss

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

# Main function to run the DQN algorithm
def main():
    env = gym.make('CartPole-v1')  # Initialize the CartPole environment
    q = Qnet()  # Initialize the Q-network
    q_target = Qnet()  # Initialize the target Q-network
    q_target.load_state_dict(q.state_dict())  # Synchronize the target network with the Q-network
    memory = ReplayBuffer()  # Initialize replay buffer

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)  # Adam optimizer

    for n_epi in range(10000):  # Run for 10000 episodes
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing of epsilon from 8% to 1%
        s, _ = env.reset()  # Reset environment and get initial state
        done = False

        episode_score = 0.0  # Initialize the episode score

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)  # Sample action using epsilon-greedy policy
            s_prime, r, done, truncated, info = env.step(a)  # Take action and get next state and reward
            done_mask = 0.0 if done else 1.0  # Mask for terminal state
            memory.put((s, a, r / 100.0, s_prime, done_mask))  # Store transition in replay buffer
            s = s_prime  # Move to the next state

            episode_score += r  # Update episode score

            if done:
                break

        score += episode_score

        if memory.size() > 2000:  # Start training after the buffer has enough samples
            train(q, q_target, memory, optimizer)

        if episode_score > 200 and episode_score < 1000:  # Save the model if the episode score is greater than 200
            torch.save(q.state_dict(), f"cartpole_model_{n_epi}.pth")
            print(f"Model saved at episode {n_epi} with score {episode_score}")

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())  # Synchronize target network
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    env.close()  # Close the environment

# Run the main function
if __name__ == '__main__':
    main()