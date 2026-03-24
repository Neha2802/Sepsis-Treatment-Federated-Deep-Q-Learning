import sys
import numpy as np
#import matplotlib.pyplot as plt
import pylab as p

def parse_round_reward(line):
    parts = line.strip().split('|')
    if len(parts) != 2:
        return None, None
    round_info = parts[0].split(' ')
    round_num = int(round_info[1])
    reward_info = parts[1].split(':')
    reward_str = float(reward_info[1])
    # If the value is an integer (e.g., "78"), convert it to int for consistency
    if isinstance(reward_str, int):
        reward = int(reward_str)
    else:
        reward = float(reward_str)
    return round_num, reward

def main():
    try:
        rounds = []
        rewards = []
        
        # Read all lines from the file
        with open('output.txt', 'r') as f:
            for line in f.readlines():
                rounded, reward = parse_round_reward(line.strip())
                if rounded is not None and reward is not None:
                    rounds.append(rounded)
                    rewards.append(reward)

        print(rounds)
        print(np.mean(rewards))
        # Plotting
        #plt.figure(figsize=(10, 6))
        p.plot(rounds, rewards)
        # plt.title('Round Number vs Average Test Reward')
        # plt.xlabel('Round')
        # plt.ylabel('Average Reward')
        # plt.grid(True)
        print("plot printed")
        p.show()

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()