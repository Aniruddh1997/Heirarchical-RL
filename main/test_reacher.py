import gym
import numpy as np
import pickle
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp_policy import Policy
from models.mlp_policy_disc import DiscretePolicy
from core.agent import get_target

env = gym.make("FetchReach-v1", reward_type='dense')
is_disc_action = len(env.action_space.shape) == 0

policy_mgr, policy_wrk, _, _, _= pickle.load(open("../assets/learned_models/FetchReach-v1_trpo_dense.p", "rb"))
state = env.reset()
total_reward = 0.0
steps = 0
direction = policy_mgr.select_action(torch.tensor(np.concatenate((state['observation'],state['desired_goal']))).unsqueeze(0))[0]
direction = int(direction.detach().numpy())
curr_pos = state['achieved_goal']
subgoal = get_target(curr_pos,direction)
#done_count = 0
for i in range(2000):

    env.render()
    state_wrk = np.concatenate((state['observation'], state['desired_goal'], subgoal))
    action = policy_wrk.select_action(torch.tensor(state_wrk).unsqueeze(0))[0]
    state, reward, done, info = env.step(action.detach().numpy())
    #print (done_count)
    #if(done): done_count += 1
    #print (done_count)
    #done = (done_count == 1000)
    total_reward += reward
    steps += 1

    reward_wrk = - np.linalg.norm(subgoal - state['achieved_goal'])
    subgoal_reached = (-reward_wrk < 0.05)    

    # print ('W:',-reward_wrk)

    if(subgoal_reached):
        direction = policy_mgr.select_action(torch.tensor(np.concatenate((state['observation'],state['desired_goal']))).unsqueeze(0))[0]
        direction = int(direction.detach().numpy())
        curr_pos = state['achieved_goal']
        subgoal = get_target(curr_pos,direction)
        # print('Manager:',np.linalg.norm(subgoal - state['desired_goal']))
        #print ("     ")

    if done:
        state = env.reset()
        #done_count = 0
        direction = policy_mgr.select_action(torch.tensor(np.concatenate((state['observation'],state['desired_goal']))).unsqueeze(0))[0]
        direction = int(direction.detach().numpy())
        curr_pos = state['achieved_goal']
        subgoal = get_target(curr_pos,direction)
        print(total_reward, steps)
        print("   ")
        # print('Manager:',np.linalg.norm(subgoal - state['desired_goal']))
        total_reward = 0.0
        steps = 0
        #break


# env = gym.make("FetchReach-v1", reward_type='dense')
# state = env.reset()
# total_reward = 0
# for i in range(2000):
#     env.render()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     total_reward += reward
#     if done: 
#         state = env.reset()
#         print(total_reward)
#         total_reward = 0


