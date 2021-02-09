import numpy as np
import torch, logging, gym
import matplotlib.pyplot as plt
import retro
from a3c_model import *
from utils import *
import os.path
from datetime import datetime, timedelta

NB_FRAMES_MAX = 8e7
NB_FRAMES_CHECKPOINT = 1e5
NB_FRAMES_PRINT = 1e3
NB_FRAMES_PLOT = 1e5

EP_LENGHT_MAX = 1e4
GAMMA = 0.99
ENTROPY_FACTOR = 0.00

def compute_actor_loss(values, rewards, actions, log_prob):
    delta_t = rewards + GAMMA * values.view(-1).data[1:] - values.view(-1).data[:-1]
    gae = torch.cuda.FloatTensor(discount_sum(delta_t.cpu().numpy(), GAMMA ).copy())
    log_prob_  = log_prob.gather(1, actions.view(-1,1)).view(-1)
    
    return -(log_prob_  * torch.cuda.FloatTensor(gae.clone())).sum()
    
def computer_critic_loss(values, rewards):
    rewards[-1] += GAMMA * values.view(-1).data[-1]
    rewards_ = torch.cuda.FloatTensor(discount_sum(np.asarray(rewards.cpu().numpy()), GAMMA).copy())
    
    return 0.5*(rewards_ - values[:-1,0]).pow(2).sum()

def compute_loss(values, log_prob, actions, rewards):
    actor = compute_actor_loss(values, rewards, actions, log_prob)
    critic = computer_critic_loss(values, rewards)
    entropy = (-log_prob * torch.exp(log_prob)).sum() 

    return actor + 0.5*critic - ENTROPY_FACTOR*entropy


def run(args, log, shared_model, shared_optim, worker_index):
    if worker_index == 0:
        rewards_log = list()
        frames_log = list()
        ep_log = list()
        loss_log = list()

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(shared_optim, 'max')

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(filename=args.log_path,level=logging.DEBUG)
    
    env = retro.make(args.env, args.envState) 
    env.seed(worker_index) 
    torch.manual_seed(worker_index) 
    shared_gradient_initialized = False
    model = A3C_Model(channels=1, hidden_gru_size=args.hidden_gru_size, num_actions=args.num_actions, use_batch_norm=args.no_batch_norm, deeper=args.deeper) # a local/unshared model
    state = torch.tensor(preprocess_frame(env.reset())) 

    ep_length = ep_reward = ep_loss = avg_speed = 0
    old_x = old_y = None 
    done = True 
    last_ep_reward = None
    local_n_frames = 0

    while True: 
        last_render = datetime.min
        values = []
        log_prob = []
        actions = []
        rewards = []
        model.load_state_dict(shared_model.state_dict())
        if done:
            gru_x = torch.zeros(1, args.hidden_gru_size)
        else:
            gru_x = gru_x.detach()  

        for i in range(args.update_frequency):
            ep_length += 1

            logits, value, gru_x = model((state.view(1,1,RES_F,RES_F), gru_x))
            values.append(value)
            log_prob.append(torch.nn.functional.log_softmax(logits, dim=-1))
            action = torch.exp(torch.nn.functional.log_softmax(logits, dim=-1)).multinomial(num_samples=1).data[0]
            actions.append(action)
            actionArray = np.zeros(args.num_actions, dtype=bool)
            actionArray[action] = 1
            state, reward, done, gamedata = env.step(actionArray)
            

            
            ep_reward += reward
            if args.no_render and worker_index == 0:
                while args.test and datetime.now() - last_render < timedelta(microseconds = 16666):
                    pass
                
                last_render = datetime.now()    
                env.render()
                

            state = torch.tensor(preprocess_frame(state))  
            if gamedata['dead'] == 0 : #penalize death and finishes 
                done = True
                reward = -10
            if gamedata['endOfLevel'] == 1:
            	reward = +10 #big reward when finishing level

            

            reward = reward + (gamedata['x'] - old_x if old_x is not None else 0) + (max(2*(old_y - gamedata['jump']) , -1) if old_y is not None else 0) # reward for going right and up, y pos reverted
            
            if old_x is not None:
                avg_speed = 0.999*(max(gamedata['x'] - old_x,0)) + 0.001*avg_speed
            if avg_speed < 0.01 :
                reward -= 0.5
                #if worker_index == 0:
                #    print('tata')
            reward/=10
            #if worker_index == 0:
            #    print(reward)
            #    print(gamedata)
            
            old_x = gamedata['x']
            old_y = gamedata['jump']
            rewards.append(reward)
            done = done or ep_length >= EP_LENGHT_MAX
            
            log['n_frame'] += 1
            local_n_frames += 1

            if done: 
                log['n_episode'] += 1
                history = None
                if log['n_episode'][0] == 1:
                    history = 1
                else:
                    history =  1 - args.average_decay
                log['ep_reward_avg'] *= 1-history
                log['ep_reward_avg'] += history * ep_reward
                log['ep_loss_avg'] *= 1-history
                log['ep_loss_avg'] += history * ep_loss
                
                #lr_scheduler.step(log['ep_reward_avg'])

                if worker_index == 0:
                    rewards_log.append(log['ep_reward_avg'].item())
                    frames_log.append(log['n_frame'].item())
                    ep_log.append(log['n_episode'].item())
                    loss_log.append(log['ep_loss_avg'].item())

                ep_length = ep_reward = ep_loss = 0
                state = torch.tensor(preprocess_frame(env.reset()))

            if worker_index == 0:
                must_save = False
                if last_ep_reward is not None:
                    if local_n_frames % NB_FRAMES_CHECKPOINT == 0 and last_ep_reward < log['ep_reward_avg']:
                        last_ep_reward = log['ep_reward_avg'].item()
                        must_save = True
                else:
                    if local_n_frames % NB_FRAMES_CHECKPOINT == 0:
                        last_ep_reward = log['ep_reward_avg'].item()
                        must_save = True
                        

                if must_save:           
                    fr = 'n_frame'; ep = 'n_episode'
                    record(f'The model has been saved. Nb of frames processed: {log[fr].item():.0f}\n')
                    torch.save(shared_model.state_dict(), os.path.join(args.checkpoint_path, f'checkpoint.{log[ep].item():.0f}.{log[fr].item()/NB_FRAMES_CHECKPOINT:.0f}.ckpt'))

                if local_n_frames % NB_FRAMES_PRINT == 0: 
                    ep = 'n_episode' ; rew = 'ep_reward_avg' ; epl = 'ep_loss_avg' ; fr = 'n_frame'
                    record(f'Total episodes: {log[ep].item():.0f} | Total processed frames: {log[fr].item():.0f} | Average reward: {log[rew].item():.2f} | Loss: {log[epl].item():.2f} ' + 'lr:' + str(shared_optim.param_groups[0]['lr']))
                    
                 
                if local_n_frames % NB_FRAMES_PLOT == 0:
                    plt.plot(ep_log, rewards_log)
                    plt.title(args.env)
                    plt.xlabel('# of episodes')
                    plt.ylabel('avg_reward')
                    plt.savefig(os.path.join(args.checkpoint_path, 'rewPerEp.png'))
                    plt.clf()

                    plt.plot(frames_log, rewards_log)
                    plt.title(args.env)
                    plt.xlabel('# of frames processed')
                    plt.ylabel('avg_reward')
                    plt.savefig(os.path.join(args.checkpoint_path, 'rewPerFr.png'))
                    plt.clf()

                    plt.plot(ep_log, loss_log)
                    plt.title(args.env)
                    plt.xlabel('# of episodes')
                    plt.ylabel('avg_loss')
                    plt.savefig(os.path.join(args.checkpoint_path, 'lossPerEp.png'))
                    plt.clf()

                    print("Plots saved\n")
        
        if done:
            next_value = torch.zeros(1,1) 
        else:
            next_value = model((state.view(1,1,RES_F,RES_F), gru_x))[1]

        values.append(next_value.detach())
        rewards = torch.from_numpy(np.array(rewards)).cuda()
        loss = compute_loss(torch.cat(values), torch.cat(log_prob), torch.cat(actions), rewards)
        ep_loss += loss.item()

        if not args.test:
            shared_optim.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 45)
            if not shared_gradient_initialized:
                sync_gradients(shared_model, model)
                shared_gradient_initialized = True
            shared_optim.step()
