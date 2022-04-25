
import gym
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch as T
import os
from util import plot_learning_curve, make_env



class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        print(input_dims[0])
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        print(fc_input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))
    
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        conv_state=T.flatten(conv3)
        
        # conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

class Agent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 eps_min=0.01, eps_dec=5e-7,algo=None, env_name=None, chkpt_dir=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.algo = algo
        self.chkpt_dir = chkpt_dir
        self.env_name = env_name
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0




        self.Q = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random()> self.epsilon:
                state= T.tensor(observation, dtype=T.float).to(self.Q.device)
                actions=self.Q.forward(state)
                action=T.argmax(actions).item()

        else:
            action=np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        self.epsilon=self.epsilon- self.eps_dec \
            if self.epsilon> self.eps_min else self.eps_min


    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()

        states=T.tensor(state, dtype=T.float).to(self.Q.device)
        actions= T.tensor(action).to(self.Q.device)
        rewards=T.tensor(reward).to(self.Q.device)
        states_=T.tensor(state_, dtype=T.float).to(self.Q.device)

        # print(states)
        # print(actions)
        # print(states.size())
        q_pred=self.Q.forward(states)[actions]

        q_next=self.Q.forward(states_).max()

        q_target=reward + self.gamma*q_next

        loss= self.Q.loss(q_target,q_pred).to(self.Q.device)

        loss.backward()

        self.Q.optimizer.step()
        self.decrement_epsilon()

    def save_models(self):
        self.Q.save_checkpoint()


if __name__== '__main__':
    
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    n_games = 250
    scores, eps_history, steps_array = [], [], []


    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.00025,
                    input_dims=(env.observation_space.shape),
                    n_actions=env.action_space.n, eps_min=0.1,
                    eps_dec=1e-6,algo='NaiveAgent',
                    env_name='PongNoFrameskip-v4',
                    chkpt_dir='models/')
    

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'


    n_steps = 0
    

    for i in range(n_games):
        done = False
        observation = env.reset()


        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation,action,reward,observation_ )
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)  
        
        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)    
        
        if avg_score > best_score:
            best_score = avg_score        

        eps_history.append(agent.epsilon)

    agent.save_models()  
    plot_learning_curve(steps_array, scores, eps_history, figure_file)












# take a random action





# action_space = [i for i in range(env.action_space.n)]

# action = np.random.choice(action_space)

# shape=env.observation_space.low.shape
# # print(*shape)

# frame_buffer=np.zeros_like((2,shape))
# env.reset()
# t_reward = 0.0
# done = False
# for i in range(4):
#     obs, reward, done, info = env.step(action)
#     t_reward += reward
#     print(reward)
#     idx = i % 2
#     frame_buffer[idx] = obs
#     if done:
#      break

# print(frame_buffer.shape)

# # state_memory = np.zeros((1, *(210,1,3)),dtype=np.float32)

# # print(state_memory)





