import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time


def collect_samples(pid, queue, env, policy_mgr, policy_wrk, custom_reward,
                    mean_action, render, running_state, min_batch_size):
    torch.randn(pid)
    log = dict()
    memory_mgr = Memory()
    memory_wrk = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    avg_wrk_reward = 0 
    avg_mgr_reward = 0 

    mgr_steps = 0
    done_count = 0
    state = env.reset()
    while num_steps < min_batch_size:
        #state_wrk = tensor(state['observation'])
        #state = np.concatenate((state['observation'],state['desired_goal']))
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        state_mgr = tensor(np.concatenate((state['observation'],state['desired_goal']))).unsqueeze(0)
        with torch.no_grad():
            direction = policy_mgr.select_action(state_mgr)[0]

        direction = int(direction.detach().numpy())
        curr_pos = state['achieved_goal']
        subgoal = get_target(curr_pos,direction)
        state_wrk = tensor(np.concatenate((state['observation'], state['desired_goal'], subgoal)))
         


        for t in range(10000):
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy_wrk.select_action(state_wrk.unsqueeze(0))[0].numpy()
            next_state, reward, done, info = env.step(action)

            # dist = np.linalg.norm(info['fingertip']-info['target'])

            next_state_wrk = np.concatenate((next_state['observation'], next_state['desired_goal'], subgoal))
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask_mgr = 0 if done else 1

            #reward_wrk = - np.linalg.norm(subgoal - next_state['achieved_goal'])
            reward_wrk = - np.linalg.norm(subgoal - next_state['achieved_goal'])
            #reward_wrk = reward
            subgoal_reached = (-reward_wrk < 0.05)
            mask_wrk = 0 if (done or subgoal_reached) else 1
            #mask_wrk = 0 if (done) else 1

            memory_wrk.push(state_wrk.detach().numpy(), action, mask_wrk, next_state_wrk, reward_wrk)
            avg_wrk_reward += reward_wrk


            if render:
                env.render()
            if (done or subgoal_reached):
            #if (done):
                break

            state_wrk = tensor(next_state_wrk)

        #next_state_mgr = np.concatenate((next_state['observation'],next_state['desired_goal']))
        next_state_mgr = next_state['observation']
        #reward_mgr = reward_episode - 10*np.linalg.norm(next_state['achieved_goal'] - next_state['desired_goal'])
        #reward_mgr = reward_episode/50.0 -  np.linalg.norm(subgoal - info['target'])
        reward_mgr = reward_episode/50.0 
        memory_mgr.push(np.concatenate((state['observation'],state['desired_goal'])), direction, mask_mgr, next_state_mgr, reward_mgr)

        state = next_state
        avg_mgr_reward += reward_mgr
        mgr_steps += 1

        # log stats
        num_steps += (t + 1)
        if(done): 
            num_episodes += 1
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
            state = env.reset()
            curr_pos = state['achieved_goal']
            total_reward += reward_episode

        else:
            curr_pos = state['achieved_goal']

    log['num_steps'] = num_steps
    log['mgr_steps'] = mgr_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / (num_episodes)
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['mgr_reward'] = avg_mgr_reward / mgr_steps
    log['wrk_reward'] = avg_wrk_reward / num_steps
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory_mgr, memory_wrk, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy_mgr, policy_wrk, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy_mgr = policy_mgr
        self.policy_wrk = policy_wrk
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy_mgr)
        to_device(torch.device('cpu'), self.policy_wrk)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy_mgr, self.policy_wrk, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory_mgr, memory_wrk, log = collect_samples(0, None, self.env, self.policy_mgr, self.policy_wrk, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size)

        #worker_logs = [None] * len(workers)
        #worker_memories = [None] * len(workers)
        #for _ in workers:
        #    pid, worker_memory, worker_log = queue.get()
        #    worker_memories[pid - 1] = worker_memory
        #    worker_logs[pid - 1] = worker_log
        #for worker_memory in worker_memories:
        #    memory.append(worker_memory)
        batch_mgr = memory_mgr.sample()
        batch_wrk = memory_wrk.sample()
        #if self.num_threads > 1:
        #    log_list = [log] + worker_logs
        #    log = merge_log(log_list)
        #to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        #log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        #log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        #log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch_mgr, batch_wrk, log



def get_target(x, dir, off = 0.1):
    # res = np.zeros(3)
    # if dir == 0:
    #     res[0] = off
    # elif dir == 1:
    #     res[0] = -off
    # elif dir == 2:
    #     res[1] = off
    # elif dir == 3:
    #     res[1] = -off
    # elif dir == 4:
    #     res[2] = off
    # elif dir == 5:
    #     res[2] = -off
    # elif dir == 6:
    #     res[0] = off
    #     res[1] = off
    #     res[2] = off
    # elif dir == 7:
    #     res[0] = -off
    #     res[1] = off
    #     res[2] = off
    # elif dir == 8:
    #     res[0] = off
    #     res[1] = -off
    #     res[2] = off
    # elif dir == 9:
    #     res[0] = off
    #     res[1] = off
    #     res[2] = -off
    # elif dir == 10:
    #     res[0] = -off
    #     res[1] = -off
    #     res[2] = off
    # elif dir == 11:
    #     res[0] = -off
    #     res[1] = off
    #     res[2] = -off
    # elif dir == 12:
    #     res[0] = off
    #     res[1] = -off
    #     res[2] = -off
    # elif dir == 13:
    #     res[0] = -off
    #     res[1] = -off
    #     res[2] = -off
    
    
    res = np.zeros(3)
    if dir == 0:
        res[0] = off
    elif dir == 1:
        res[0] = -off
    elif dir == 2:
        res[1] = off
    elif dir == 3:
        res[1] = -off
    elif dir == 4:
        res[2] = off
    elif dir == 5:
        res[2] = -off
    # elif dir == 6:
    #     res[0] = off
    #     res[1] = off
    # elif dir == 7:
    #     res[0] = off
    #     res[1] = -off
    # elif dir == 8:
    #     res[0] = -off
    #     res[1] = off
    # elif dir == 9:
    #     res[0] = -off
    #     res[1] = -off
    # elif dir == 10:
    #     res[0] = off
    #     res[2] = off
    # elif dir == 11:
    #     res[0] = off
    #     res[2] = -off
    # elif dir == 12:
    #     res[0] = -off
    #     res[2] = off
    # elif dir == 13:
    #     res[0] = -off
    #     res[2] = -off
    # elif dir == 14:
    #     res[1] = off
    #     res[2] = off
    # elif dir == 15:
    #     res[1] = off
    #     res[2] = -off
    # elif dir == 16:
    #     res[1] = -off
    #     res[2] = off
    # elif dir == 17:
    #    res[1] = -off
    #    res[2] = -off
    
    return x + res
