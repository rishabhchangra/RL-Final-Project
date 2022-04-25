import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', test=False):
        
        #discount factor
        self.gamma = gamma
        #epsilon
        self.epsilon = epsilon
        #learning rate for Optimizer
        self.lr = lr
        #number of actions
        self.n_actions = n_actions
        #input dimensions
        self.input_dims = input_dims
        #batch size 
        self.batch_size = batch_size
        #Minimum epsilon value
        self.eps_min = eps_min
        #Epsilon decay
        self.eps_dec = eps_dec
        #replace target network frequency
        self.replace_target_cnt = replace
        #algorithm name
        self.algo = algo
        #env name
        self.env_name = env_name
        #checkpoint dir path
        self.chkpt_dir = chkpt_dir
        #creating a list of actions
        self.action_space = [i for i in range(n_actions)]
        #step counter 
        self.learn_step_counter = 0

        self.test=test


        #instantiating memory buffer for storing the experiences
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        #creating the Evaluation Network to approximate Q values
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)
        #creating the Target Network to calculate the next Q values 
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    
    
    def choose_action(self, observation):
        
        #when we are performing test, we choose the maximum action
        if self.test:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
            return action        
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        #storing the experience tuple
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        #sampling the experience tuple
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)
        
        #converting the tuple values into tensors 
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        #replacing the target network with eval network weights after periodically
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        ##decrementing epsilon after every learning step
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):

        #when memory counter is less than the batch size, function returns and no learning is performed
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
         #calling replace target function
        self.replace_target_network()

        
        #sampling experience from the buffer
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        #Estimation of the Q value using the evaluation network
        q_pred = self.q_eval.forward(states)[indices, actions]

        #Estimation of the Q value for next state using the Target network and finding the max action
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        #setting q_next to zero for the terminating states 
        q_next[dones] = 0.0

        #calculating the target value of q using gamma,reward and max q_next 
        q_target = rewards + self.gamma*q_next

        #calculating the loss and backpropagating 
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        #decrementing the epsilon
        self.decrement_epsilon()
