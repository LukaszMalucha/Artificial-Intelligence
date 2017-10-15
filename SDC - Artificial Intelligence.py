## A.I. "Brain" for Self-Driving Car - selecting right action each time

## Importing the libraries

import numpy as np
import random                               ## for random samples
import os                                   ## for save/load
import torch                                ## pytorch neural network with graphs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim                 ## for optimizers
import torch.autograd as autograd           ## converts tensor into variable with gradient
from torch.autograd import Variable

################################################## Neural Network Architecture ##########################################################################

class Network(nn.Module):                   ## inherit from parent class

        def __init__(self, input_size, nb_action ):             ## self, 5 neurons to describe environment, 3 actions
                super(Network, self).__init__()                 ## trick to use all the tools of nn.Module
                self.input_size = input_size                    ## getting input if specified 
                self.nb_action = nb_action                      ## getting input if specified
                self.fc1 = nn.Linear(input_size,                ## full connections between different layers - all input neurons to hidden layer
                                     30)                        ## 30 neurons as EXPERIMENTAL value - can be altered
                self.fc2 = nn.Linear(30, nb_action)             ## second full connection between hidden and output layer
         

### Function for Forward Propagation path:
        
        def forward(self, state):                               ## state for input entering the neural network
                x = F.relu(self.fc1(state))                     ## activate hidden neurons with rectifier (line 23)
                q_values = self.fc2(x)                          ## q-values for each of our actions
                return q_values                                 ## return value for: "go right, go straight or go left"

### Impelementin Experience Replay to understand long-term correlations - MEMORY of last 100 events:

class ReplayMemory(object):
        
        def __init__(self, capacity):
                self.capacity = capacity                        ## how many events should be memorized
                self.memory = []                                ## initializing memory   
        
        def push(self, event):                                  ## transition of element to the memory
                self.memory.append(event)                       ## append new event to memory list
                if len(self.memory) > self.capacity:            ## make sure that memory doesn't grow over capacity
                        del self.memory[0]
        
        def sample(self, batch_size):                           ## gather random sample from event memory
                sample = zip(*random.sample(self.memory, batch_size))          ## zip to group varaibles of each element (first with first, second with second etc...) 
# takes the samples, concatinate them with respect to first dimension(0), and converts into torch variables (tensor + gradient)             
                return map(lambda x: Variable(torch.cat(x,0)), samples)            ## list of batches where each of them is torch variable
               
                
        
###################################################  Implementing Deep-Q model ##############################################################################
        
        
class DQN():
        
        def __init__(self, input_size, nb_action, gamma):      
                self.gamma = gamma                                             ## delay coefficient
                self.reward_window = []                                        ## rewards in a list
                self.model = Network(input_size, nb_action)
                self.memory = ReplayMemory(100000)    
                self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)           ## optimzier for learning rate
                ## last state need to be torch tensor with one extra 'fake' dimension corresponding with batch number (has to be first dim):
                self.last_state = torch.Tensor(input_size).unsqueeze(0)                     ## initialize Tensor 
                self.last_action = 0                                           ## corresponding to action2rotation angle from "SDC_Environment"
                self.last_reword = 0                                           ## initializing 
                
        
## Select right action each time:
        
        def select_action(self, state):
                probs = F.softmax(self.model(Variable(state,                   ## softmax - sum of q values = 1
                                                      volatile = True,         ## gradient will be incorporated into graph
                                                      ))*100)                    ## T=7 for temperature of confidence (higher then less insect-like). Zero to deactivate A.I.  
                action = probs.multinomial()                                   ## random draw from distribution
                return action.data[0,0]
        
## Train DQ Network with backpropagation:

        def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
                ## unsqueeze to add missing 'fake' dimension to batch.action and then kills them with sqeeze to get back to oiginal form after
                outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)      ## specify only action that was chosen with 'gather'
                next_outputs = self.model(batch_next_state).detach().max(1)[0]                         ## get maximum with detach --> max
                target = self.gamma*next_outputs + batch_reward                                        ## as per mathematical definition
                td_loss = F.smooth_l1_loss(outputs, target)                                            ## error of the prediction - best option for deep-q
                self.optimizer.zero_grad()                                                             ## initialize optimizer with each iteration of the loop
                td_loss.backward(retain_variables = True)                                              ## BACKPROPAGATION to the netowrk
                self,optimizer.step()                                                                  ## update the weights
        
                
## Update all the elements of a transition and returns action that was played:
        
         def update(self, reward, new_signal):                                 ## as in Environment - line 131
                 new_state = torch.Tensor(new_signal).float().unsqueeze(0)     ## transform signal into tesnor with extra dim
                 self.memory.push((self.last_state, new_state,                 ## update the memory with push line 43
                                   torch.LongTensor([int(self.last_action)]),  ## TRANSITION CAN ONLY CONTAIN TENSORS
                                   torch.Tensor([self.last_reward])))          ## reward update
                 action = self.select_action(new_state)                        ## Defining new state after transition - play new action 
                 if len(self.memory.memory) > 100:                             ## memory from line 41
                         batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) ## learning from last 100 samples
                         self.learn(batch_state, batch_next_state, batch_reward, batch_action)
                 self.last_action = action
                 self.last_state = new_state
                 self.last_reward = reward
                 self.reward_window.append(reward)
                 if len(self.reward_window) > 1000:
                         del self.reward_window[0]
                 return action
         
## Compute the score:
        
        def score(self):
                return sum(self.reward_window)/(len(self.reward_window)+1.)    ## +1 as a safety measure for denominator

## Save parameters of the model:    
        def save(self):
                torch.save({'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict},
                         'last_brain.pth')     
                
## Load function
        def load(self):
                if os.path.isfile('last_brain.pth'):
                        print("=> loading checkpoint... ")
                        checkpoint = torch.load('last_brain.pth')
                        self.model.load_state_dict(checkpoint['state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        print("done !")
                else:
                        print("no checkpoint found...")                       

                          
                                   
                                   
                                   

        
                                              
                                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        