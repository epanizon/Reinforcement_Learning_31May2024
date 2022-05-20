import numpy as np
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import gym
from gym import wrappers
from utils import * 
from main_classes import * 
import matplotlib.pyplot as plt
import numpy as np
from collections import deque 
from garage.envs.wrappers import PixelObservationWrapper, Grayscale, Resize,StackFrames
import torchvision.transforms as transform
from ut import *

if __name__ == '__main__': 

    disable_view_window()

    env = gym.make('BipedalWalkerHardcore-v3')
    
    env = PixelObservationWrapper(env)    
    
    disable_view_window()
    
    env = Grayscale(env)

    env = Crop_Observation(env)
    
    env= Resize(env, 250, 250) 
    
    env = StackFrames(env, 3, 0)

    
    s = env.reset() 
    
    #mean = 0.8300 
    
    #std = 0.1782 
    
   # transforms = transform.Compose([
    #T.tensor(s,dtype=T.float)
   # transform.Normalize([mean,mean,mean,mean], [std,std,std,std]),
   # ])
    dev = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
   # dev ="cpu"
   
    agent = Agent(env, 0.001,0.001,device = dev,name="pixels_and_values_model_without_lstm",chkpt="model_1")
    
    #agent = Agent(env,0.0002,0.0001,transforms=transforms,device = dev,name="prova_CCN")
    #agent = Agent(env,0.005,0.005,0.005,transforms=transforms,device=dev,name="prima_prova_CCN")
    EPISODES = 20
    filename = "model_1" 
    best_score = env.reward_range[0]
    score_history = [] 
    step = 0 
    expl_noise = 0.1
    #env.close() 
    
    for i in range(1, EPISODES+1): 
        #expl_noise *= 0.999
    
        state = env.reset() 
    
        done = False 
    
        score = 0 
    
        while not done: 
        
            #print(step)
        
            #step+=1
           
        
            action = agent.choose_action(state,expl_noise)
            
            new_state, reward, done, _ = env.step(action[0]) 
           
            if reward<=-100: 
                reward = -1
               # print(state.shae)
                agent.remember(state, action, reward, new_state, True)
            else: 
                agent.remember(state, action, reward, new_state, False)
                
                
        
            agent.learn() 
        
            score += reward 
        
            state = new_state 
        
        score_history.append(score)
    
        avg_score = np.mean(score_history[-20:])
    
        if avg_score > best_score: 
        
            best_score = avg_score 
        
        if (avg_score == best_score or i%10==0):
        
            agent.save_models() 
        
        print("Results during training procedure : ")  
         
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)
          
        
    
    
            
    x = [i+1 for i in range(EPISODES)]    

    figure_file = "model_1.png"

    plot_average_reward(x, score_history,figure_file)
    
    with open('model_1.txt','w') as f: 
    
       for i in range(0,len(score_history)): 
        
           f.write(str(score_history[i])+"\n") 
        

    
    
    
    
    

    
    
    
        
    
    
    
    
   
