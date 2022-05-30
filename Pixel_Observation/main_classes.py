import numpy as np
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import gym
from gym import wrappers
from utils import * 

import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque 
from garage.envs.wrappers import PixelObservationWrapper, Grayscale, Resize,StackFrames
import os 
import torchvision.transforms as transform
from ut import *

MAX_ACTION = gym.make('BipedalWalkerHardcore-v3').action_space.high[0] 

class Crop_Observation(gym.ObservationWrapper): 
    
    def __init__(self,env):
        super().__init__(env) 
        
    def observation(self, obs): 
        return crop_image(obs)
        

class Actor(nn.Module): 

    def __init__(self, n_actions, in_channels=3, name = None, chkpt = "prova_lstm"): 
    
        super(Actor, self).__init__() 
        #self.actions = n_actions
        self.name = name 
        
        if name is not None: 
        
            if not os.path.exists(chkpt): 
            
                os.makedirs(chkpt) 
                
            self.filename = os.path.join(chkpt, name +'_ddpg') 
            
        self.in_channels = in_channels 
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
             
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
             
        self.conv3 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
            
             
        self.conv4 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
            
             
        self.conv5 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 1),
            
            nn.BatchNorm2d(16),
            
            ) 
        
        self.max_pool = nn.MaxPool2d(kernel_size =(2,2),stride=(2,2))
            
        #self.linear1 = nn.Linear(16*6*6,64)
        
        #self.linear2= nn.Linear(64, 128)
        
        #self.linear3 = nn.Linear(128, n_actions)
        
        #self.lstm = nn.LSTMCell(16, 32)
            
        self.linear1 = nn.Linear(16,64)
        
        self.linear2 = nn.Linear(64,128)
        
        #self.linear2= nn.Linear(64, 128)
        
        self.linear3 = nn.Linear(128, n_actions)
        
        
        
    def forward(self, x): 
    
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        x = F.relu(self.conv4(x))
        x = self.max_pool(x) 
        x = F.relu(self.conv5(x))
        x = self.max_pool(x)
        x = x.reshape(x.shape[0],-1)
    
        x = F.relu(self.linear1(x))
        
        x = F.relu(self.linear2(x))
        
        #x = T.tanh(self.linear3(x))*MAX_ACTION
        
       # x = F.adaptive_avg_pool2d(x,1)
        #x = x.view(-1, 16 * 1 * 1)
       # x,y = self.lstm(x)
    
       # x = F.relu(self.linear1(x))
        
       # x = F.relu(self.linear2(x))
        
        x = T.tanh(self.linear3(x))*MAX_ACTION
        
        return x 
        
    def save_checkpoint(self):
        
        if self.name is not None:
            
            print("saving")
        
            T.save(self.state_dict(), self.filename)
        

    def load_checkpoint(self):
        
        if self.name is not None:
        
            print("loading")
    
            self.load_state_dict(T.load(self.filename))
            
            

class Critic(nn.Module): 
    
    def __init__(self, n_actions, in_channels=3, name = None, chkpt = "pixels_and_values"): 
        
        super(Critic, self).__init__() 
        
        self.name = name 
        
        if name is not None: 
            
            if not os.path.exists(chkpt): 
                
                os.makedirs(chkpt) 
                
            self.filename = os.path.join(chkpt,name +'_ddpg') 
            
        self.in_channels = in_channels
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels,16,3, padding =1, stride=2),
            
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
             
        self.conv3 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
            
             
        self.conv4 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 2),
            
            nn.BatchNorm2d(16),
            
            ) 
            
            
             
        self.conv5 = nn.Sequential(
            
            nn.Conv2d(16,16,3, padding = 1, stride = 1),
            
            nn.BatchNorm2d(16),
            
            ) 
        self.max_pool = nn.MaxPool2d(kernel_size =(2,2),stride=(2,2))
        
        #self.lstm = nn.LSTMCell(16, 32)
        
        self.linear1 = nn.Linear(16+n_actions,64)
        
        self.linear2= nn.Linear(64, 128)
        
        self.linear3 = nn.Linear(128, 1)
        
    
    def forward(self,state,actions):
        
        #pixel_observation, state = observation
        
        x = F.relu(self.conv1(state))
        
        
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        
        x = F.relu(self.conv3(x))
        x= self.max_pool(x)
        x = F.relu(self.conv4(x))
        x= self.max_pool(x)
        x = F.relu(self.conv5(x))
        x= self.max_pool(x)
       # x = F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.shape[0],-1)
       # x = x.view(-1, 16 * 1 * 1)
        
     
        
        out_ = T.cat([x,actions],1)
        
        x = F.relu(self.linear1(out_))
        
       # print("x.shape =",x.shape)
       # print("action sshape",actions.shape)
        
        #out_ = T.cat([x, actions],1)
        
        x = F.relu(self.linear2(x))
        
        x = self.linear3(x)
        
        return x
        
    def save_checkpoint(self):
        
        if self.name is not None:
            
            print("saving")
        
            T.save(self.state_dict(), self.filename)
        

    def load_checkpoint(self):
        
        if self.name is not None:
        
            print("loading")
    
            self.load_state_dict(T.load(self.filename))
            
            
            
class ReplayBuffer(): 

    def __init__(self, max_size, input_shape, n_actions): 
        
        self.mem_size = max_size 
        
        self.mem_cntr = 0 
        
        self.state_memory = np.zeros((self.mem_size, *input_shape)) 
        
        self.new_state_memory = np.zeros((self.mem_size, *input_shape)) 
        
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        
        self.reward_memory = np.zeros(self.mem_size) 
        
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
    def store_transition(self, state, action, reward, state_, done): 
        
        index = self.mem_cntr%self.mem_size 
        
        self.state_memory[index] = state
        
        self.new_state_memory[index] = state_ 
        
        self.terminal_memory[index] = done 
        
        self.reward_memory[index] = reward 
        
        self.action_memory[index] = action 
        
        self.mem_cntr +=1 
        
    def sample_buffer(self, batch_size): 
    
        max_mem = min(self.mem_cntr, self.mem_size) 
        
        batch = np.random.choice(max_mem, batch_size) 
        
        states = self.state_memory[batch] 
        
        states_ = self.new_state_memory[batch] 
        
        actions = self.action_memory[batch] 
        
        rewards = self.reward_memory[batch] 
        
        dones = self.terminal_memory[batch] 
        
        return states, actions, rewards, states_, dones 
    
    def __len__(self): 
        
        return self.mem_cntr
        
        


class Agent: 
    
    def __init__(self, env, critic_lr, actor_lr, tau=0.005, transforms=None, gamma=0.99,device="cpu", update_actor_interval=2, warmup=1000,max_size=10000, batch_size=64,name=None,chkpt="model",noise=0.1): 
        
        self.env = env  
        
        self.tau = tau
        
        self.transforms = transforms 
        
        self.input_dim = env.observation_space.shape 
        
        self.n_actions = env.action_space.shape[0]
        
       # self.pixel_shape = pixel_shape
        
        self.gamma = gamma 
        
        self.max_action = env.action_space.high[0] 
        
        self.min_action = env.action_space.low[0] 
        
        self.memory= ReplayBuffer(max_size, self.input_dim, self.n_actions)
        
        self.batch_size = batch_size 
        
        self.warmup = warmup 
        
        self.noise = noise 
        
        self.learn_step_cntr = 0
        
        self.time_step = 0 
        
        self.update_actor_interval = update_actor_interval 
        
        self.device = device
        
        self.name_actor = None 
        
        self.name_critic_1 = None 
        
        self.name_critic_2 = None 
        
        self.name_target_actor = None 
        
        self.name_target_critic_1 = None 
        
        self.name_target_critic_2 = None 
        
        if name is not None: 
            
            self.name_actor = name + "_actor" 
            
            self.name_critic_1 = name +"_critic_1"
            
            self.name_critic_2 = name +"_critic_2"
            
            self.name_target_actor = name + "_target_actor"
            
            self.name_target_critic_1 = name +"_target_critic_1" 
            
            self.name_target_critic_2 = name +"_target_critic_2" 
            
            
            
        self.actor = Actor(self.n_actions,name=self.name_actor,chkpt=chkpt).to(self.device)
        
        self.target_actor = Actor(self.n_actions,name=self.name_target_actor,chkpt=chkpt).to(self.device)
        
        self.critic_1 = Critic(self.n_actions,name=self.name_critic_1,chkpt=chkpt).to(self.device)
        
        self.critic_2 = Critic(self.n_actions,name=self.name_critic_2,chkpt=chkpt).to(self.device)
        
        self.target_critic_1 = Critic(self.n_actions,name=self.name_target_critic_1,chkpt=chkpt).to(self.device)
        
        self.target_critic_2 = Critic(self.n_actions,name=self.name_target_critic_2,chkpt=chkpt).to(self.device)
     
        #self.actor.initialize_weight() 
        
        #self.critic_1.initialize_weight() 
        
      #  self.critic_2.initialize_weight()
        
        self.update_network_parameters(tau=1)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr= actor_lr,weight_decay=0.01)
        
        self.critic_1_optimizer = T.optim.Adam(self.critic_1.parameters(),lr=critic_lr)
        
        self.critic_2_optimizer = T.optim.Adam(self.critic_2.parameters(),lr=critic_lr)
        
        
    def update_network_parameters(self, tau = None): 
    
        if tau is None: 
         
            tau = self.tau
        
        actor_params = self.actor.named_parameters() 
        
        critic_1_params = self.critic_1.named_parameters()
        
        critic_2_params = self.critic_2.named_parameters()
        
        
        target_actor_params = self.target_actor.named_parameters()
        
        target_critic_1_params = self.target_critic_1.named_parameters()
        
        target_critic_2_params = self.target_critic_2.named_parameters()
        

        critic_1 = dict(critic_1_params)
        
        critic_2 = dict(critic_2_params)
        
        actor = dict(actor_params)
        
       # target_critic_dict = dict(target_critic_params)
        target_actor = dict(target_actor_params)
        
        target_critic_1 = dict(target_critic_1_params)
        
        target_critic_2 = dict(target_critic_2_params)
         
        
        
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone()+ \
                                      (1-tau)*target_critic_1[name].clone()

        self.target_critic_1.load_state_dict(critic_1,strict=False)
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone()+ \
                                      (1-tau)*target_critic_2[name].clone()

        self.target_critic_2.load_state_dict(critic_2,strict=False)
        
        
        for name in actor:
            actor[name] = tau*actor[name].clone()+ \
                                      (1-tau)*target_actor[name].clone()
                                      
        self.target_actor.load_state_dict(actor,strict=False)
        
        
    def choose_action(self, state,expl_noise): 
        
        if self.time_step <= self.warmup: 
            
            mu = T.tensor(np.random.normal(scale = expl_noise, size = (self.n_actions,))).unsqueeze(0)
            
           
            
            
        else: 
            
            self.actor.eval() 
            
            
            
            state =(T.tensor(state,dtype=T.float)).to(self.device)
            
            #state = state.permute(2,0,1)
            
            if self.transforms is not None: 
                
                state = self.transforms(state)
            
            state = state.unsqueeze(0)
            
            with T.no_grad():
            
                mu = self.actor.forward(state)
            
            #mu = mu +T.tensor(np.random.normal(scale = self.noise), dtype = T.float)
            mu = mu +T.tensor(np.random.normal(0, self.max_action*expl_noise,size=self.n_actions), dtype = T.float).to(self.device) 
            
        mu_prime = T.clamp(mu, self.min_action, self.max_action)
        
        self.time_step +=1 
        
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done): 
    
        self.memory.store_transition(state, action, reward, new_state, done) 
        
    def learn(self): 
        
        if self.memory.mem_cntr < self.batch_size: 
        
            return 
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size) 
        
        reward = T.tensor(np.array(reward), dtype= T.float).to(self.device)
        
        done = T.tensor(np.array(done)).to(self.device)
        
        state = (T.tensor(np.array(state), dtype= T.float)).to(self.device)
        
        action = T.tensor(np.array(action), dtype= T.float).to(self.device)
        
        new_state = (T.tensor(np.array(new_state), dtype= T.float)).to(self.device)
        
       # state = state.permute(0,3,1,2)
        
       # new_state = new_state.permute(0,3,1,2)
        
        if self.transforms is not None: 
            
            state = self.transforms(state)
            
            new_state = self.transforms(new_state)
            
        self.target_actor.eval() 
        self.target_critic_1.eval() 
        self.target_critic_2.eval()
        target_actions = self.target_actor.forward(new_state) 
        noise = T.clamp(T.randn_like(action)*0.2*self.max_action,0.5*self.min_action,0.5*self.max_action)
        target_actions = target_actions + noise
        
       # target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)),-0.5,0.5)
        
        target_actions = T.clamp(target_actions, self.min_action, self.max_action) 
        
        Q_tc1= self.target_critic_1.forward(new_state,target_actions)
        
        Q_tc2 = self.target_critic_2.forward(new_state,target_actions) 
        self.critic_1.train() 
        self.critic_2.train()
        Q1 = self.critic_1.forward(state,action) 
        
        Q2 = self.critic_2.forward(state,action) 
        
        Q_tc1[done] = 0.0 
        
        Q_tc2[done] = 0.0 
        
        Q_tc1 = Q_tc1.view(-1) 
        
        Q_tc2= Q_tc2.view(-1) 
        
        critic_target_value = T.min(Q_tc1,Q_tc2) 
        
        target = reward +self.gamma*critic_target_value
        
        target = target.view(self.batch_size,1) 
        
        self.critic_1_optimizer.zero_grad() 
        
        self.critic_2_optimizer.zero_grad() 
        
        q1_loss = F.mse_loss(Q1,target) 
        
        q2_loss = F.mse_loss(Q2,target) 
        
        critic_loss = q1_loss + q2_loss 
        
        critic_loss.backward() 
        
        self.critic_1_optimizer.step() 
        
        self.critic_2_optimizer.step() 
        self.critic_1.eval() 
        self.critic_2.eval()
        
        self.learn_step_cntr +=1 
        
        if self.learn_step_cntr % self.update_actor_interval != 0:
            
            return 
        
        self.actor.train()
        self.actor_optimizer.zero_grad() 
        
        
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        
        actor_loss = -T.mean(actor_q1_loss) 
        
        actor_loss.backward() 
        
        self.actor_optimizer.step() 
        
        self.actor.eval()
        
        self.update_network_parameters() 
        
    def save_models(self):
        print("saving model")
        
        self.actor.save_checkpoint()
        
        self.target_actor.save_checkpoint() 
        
        self.critic_1.save_checkpoint() 
        
        self.critic_2.save_checkpoint() 
        
        self.target_critic_1.save_checkpoint() 
        
        self.target_critic_2.save_checkpoint() 
        
    def load_models(self): 
        
        self.actor.load_checkpoint() 
        
        self.target_actor.load_checkpoint() 
        
        self.critic_1.load_checkpoint() 
        
        self.critic_2.load_checkpoint() 
        
        self.target_critic_1.load_checkpoint() 
        
        self.target_critic_2.load_checkpoint() 
        
        
        
        
        
            

        
        
        
        
        
    
        

    
