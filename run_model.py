# Use this to test the pretrained model!
import gym
import numpy as np
import time
from DQN import image_processing, dnq_agent

def run_carRacing_asAgent():
    env = gym.make('CarRacing-v0').env
    while True:
        env.reset()

        total_reward = 0.0
        steps = 0
        t1 = time.time()  # Trial timer
        action = (0,0,0)
        while True:
            
            state, reward, done, info = env.step(action)
            env.render()

            procesed_image = image_processing(state) 

            action_idx = agent.make_best_move(procesed_image)
            action = agent.possible_actions[action_idx]

            # time.sleep(1/10)  # Slow down to 10fps for us poor little human!
            total_reward += reward
            if steps % 200 == 0 or done:
                print("Step: {} | Reward: {:+0.2f}".format(steps, total_reward), "| Action:", a)
            steps += 1
           
            if done:
                t1 = time.time()-t1
                scores.append(total_reward)
                scores.append(total_reward)
                print("Trial", len(scores), "| Score:", total_reward, '|', steps, "steps | %0.2fs."% t1)
                break
        env.close()


if __name__ == "__main__":
    scores = []  # Your gaming score
    a = np.array( [0.0, 0.0, 0.0] )  # Actions
    prev_err = 0

    agent = dnq_agent(epsilon=0.2,n=100,gamma=0.5)
    agent.load("/Users/Oliver/Downloads/episode_240.h5")
    print("---> MODEL LOADED!")

    run_carRacing_asAgent()
