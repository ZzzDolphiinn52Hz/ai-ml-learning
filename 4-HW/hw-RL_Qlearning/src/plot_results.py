"""Visualize training statistics after the GUI writes training_log.csv."""
import csv
import matplotlib.pyplot as plt

with open('training_log.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit('training_log.csv is empty. Train the robot first.')

episodes = [int(row['episode']) for row in rows]
rewards = [float(row['reward']) for row in rows]
success = [int(row['success']) for row in rows]
window = 100


def moving_average(values):
    return [sum(values[max(0, i-window+1):i+1]) / min(window, i+1)
            for i in range(len(values))]

plt.figure(figsize=(9, 4))
plt.plot(episodes, moving_average(rewards), label='Moving average reward (100)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-learning training progress')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('reward_curve.png', dpi=160)
print('Saved reward_curve.png')

plt.figure(figsize=(9, 4))
plt.plot(episodes, [100*s for s in moving_average(success)], label='Success rate (100)')
plt.xlabel('Episode')
plt.ylabel('Success rate (%)')
plt.ylim(0, 105)
plt.title('Robot success rate during training')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('success_curve.png', dpi=160)
print('Saved success_curve.png')
