import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

#class for DDQN agent
class DDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=10, algo=None, env_name=None, chkpt_dir='tmp/dqn', test=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.test=test

        #creating a object memory of Replay buffer class   
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

    def choose_action(self, observation):

        #when we are performing test, we choose the maximum action
        if self.test:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
            return action
        #epsilon greedy policy
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        #replacing the target network with eval network weights after periodically

        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        #decrementing epsilon after every learning step
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):

        #when memory counter is less than the batch size, function returns and no learning is performed
        
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        #calling replace target function
        self.replace_target_network()

        #sampling memory from the buffer
        states, actions, rewards, states_, dones = self.sample_memory()

        #creating a list of indices for the batch size #0 to 31 in our case
        indices = np.arange(self.batch_size)
        
       #Estimation of the Q value using the evaluation network
        q_pred = self.q_eval.forward(states)[indices, actions]
     
     
        #Estimation of the Q value for next state using the Target network
        q_next = self.q_next.forward(states_)
        #Estimation of the Q value for next state using the Evaluation network to find the max action
        q_eval = self.q_eval.forward(states_)

        #finding the max action
        max_actions = T.argmax(q_eval, dim=1)
        
        #setting q_next to zero for the terminating states 
        
        q_next[dones] = 0.0

        #calculating the target value of q using gamma,reward and max q_next where action was found using q_eval
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        #calculating the loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        
        #backpropagating the loss
        
        loss.backward()


        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
