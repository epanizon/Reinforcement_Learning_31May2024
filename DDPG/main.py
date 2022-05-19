import numpy as np 
import torch 
import gym 
import matplotlib.pyplot as plt 
from utils import *
from main_classes import * 
import os 
import argparse 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser() 
    
    parser.add_argument('critic_lr', type = float) 
    
    parser.add_argument('actor_lr', type = float) 
    
    parser.add_argument('save_dir')
    
    parser.add_argument('noise',type = int)
    
    args = parser.parse_args() 
    
    env = gym.make('BipedalWalker-v3')
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DDPGagent(env,400,300,critic_lr= args.critic_lr,actor_lr =args.actor_lr,name_critic = "critic_ddpg",name_actor="actor_ddpg",device = dev,directory = args.save_dir)
    
    EPISODES = 2
    
    name = "average_reward.png"
    
    if not os.path.exists(args.save_dir): 
    
        os.makedirs(args.save_dir)
        
    filename = os.path.join(args.save_dir,name)
    
    data_name = os.path.join(args.save_dir,"data.txt")
    
    best_score = env.reward_range[0]
    
    score_history = [] 
    
    normal_scalar = 0.24
    
    if(args.noise==0): 
    
        noise =  OUNoise(env.action_space) 
        
    step = 0
        
    for i in range(EPISODES): 
    
        state = env.reset() 
        
        done = False 
        
        score = 0 
        
        while not done: 
        
            action = agent.get_action(state) 
            
            if(args.noise==0): 
            
                action = noise.get_action(action, step) 
                
                step +=1
              
            else : 
            
                action += np.random.randn(env.action_space.shape[0])*normal_scalar 
                
                normal_scalar *= 0.9987
                
                
            new_state, reward, done, _ = env.step(action) 
            
            if reward <= -100: 
            
                reward = -1 
                
                agent.update_replay_memory(state, action, reward, new_state, True)
                
            else: 
            
                agent.update_replay_memory(state, action, reward, new_state, done) 
                
            agent.train() 
            
            score += reward 
            
            state = new_state 
            
        score_history.append(score) 
        
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score: 
        
            best_score = avg_score 
            
            agent.save_model() 
        
        if (i%50==0): 
        
            print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
            
            
        
    x = [i+1 for i in range(EPISODES)] 
    
    plot_average_reward(x,score_history,filename)
    
    with open(data_name,'w') as f: 
    
        for i in range(0,len(score_history)): 
        
            f.write(str(score_history[i])+"\n")             
        
            
        
        
            
                
                
                
                
                
         
                
                
            
                
            
                
        
        
    
    
    
    
    
         
    
    
    
    
    
    
    
    
     
    
    

    

