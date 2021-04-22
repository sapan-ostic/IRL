import numpy as np
import ptan
import torch
from torch import nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from copy import deepcopy

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class GRL():
    def __init__(self, env, noise=0.02, GAMMA=0.99):
        super(GRL, self).__init__()
        self.env = env
        self.noise = noise
        self.GAMMA = GAMMA
        self.dt = 0.01  # Discrete time step

    def calc_qvals(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.GAMMA
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))        

    def control(self, s, timer, dt=0.01):
        Kp = -100   # Controller gains 
        Kd = -3
        
        # Desired next states
        xdes = timer + self.dt
        ydes = 0.5*np.sin(6.28*(timer+self.dt))
                
        ex = s[0] - xdes
        ey = s[1] - ydes
        ex_dot = s[2]
        ey_dot = s[3]
        ax = np.array(np.round(Kp*ex + Kd*ex_dot))
        ay = np.array(np.round(Kp*ey + Kd*ey_dot))
        
        np.clip(ax, -1, 1, out=ax)
        np.clip(ay, -1, 1, out=ay)
        return (ax, ay) 

    def encode_action(self, ax, ay):
        if ax == -1 and ay == -1:
            action = 0
        elif ax == -1 and ay == 0:
            action = 1
        elif ax == -1 and ay == 1:
            action = 2
        elif ax == 0 and ay == -1:
            action = 3
        elif ax == 0 and ay == 0:
            action = 4
        elif ax == 0 and ay == 1:
            action = 5
        elif ax == 1 and ay == -1:
            action = 6
        elif ax == 1 and ay == 0:
            action = 7
        elif ax == 1 and ay == 1:
            action = 8
        return action

    def get_demonstrations(self, Ndemo=50):
        demo_states = []
        demo_actions = []
        XStore_steps = []
        AStore_steps = []
        
        idemo = 0
        while idemo is not Ndemo:    
            state = deepcopy(self.env.reset())
            
            # Storage variables
            XStore = [] # State trajectory
            AStore = [] # initial action 1, 1 # Action trajectory
            TStore = []  # Time trajectory
        
            # Initializations
            done = False 
            timer = 0

            while not done:
                timer += self.dt
                
                # Add noise to the estimation 
                state[1] += np.random.normal(-self.noise, self.noise, 1)
                
                # Get action
                ax, ay = self.control(state, timer)
                action = self.encode_action(ax, ay)
                next_state, reward, done, info = self.env.step(action)                
                
                # Store variables                
                XStore.append(state) 
                AStore.append(action)
                TStore.append(timer)
        
                state = deepcopy(next_state)
            
            # If demonstration is not perfect, ignore current trajectory   
            if info=='out_of_margin':
                continue
            
            # Todo: Fix this later 
            # XStore[0] = self.env.reset()

            # Append transitions into the list: Preprocessed data
            XStore_steps += XStore
            AStore_steps += AStore
            
            demo_states.append(XStore)
            demo_actions.append(AStore)
            
            plt.plot(np.array(XStore)[:,0], np.array(XStore)[:,1], '#aed6f1')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Demonstrations')
            idemo += 1

        keys = ['states', 'actions']
        demonstrations = dict.fromkeys(keys, None)
        demonstrations['states'] = demo_states
        demonstrations['actions'] = demo_actions
        print("Total number of demonstrations: ", len(demonstrations['states']))
        print("Total number of step demonstrations: ", len(AStore_steps))
        return demonstrations, XStore_steps, AStore_steps 

    def test_demonstrations(self, demonstrations, Nsamp=10):
        print("Testing demonstrations")
        Ndemo = len(demonstrations['actions'])
        AvgReward = 0
        for i in range(0, Ndemo, int(Ndemo/Nsamp)):
            state = self.env.reset()
            Reward = 0
            done = False
            j = 0
            while not done:
                self.env.render()
                action = demonstrations['actions'][i][j]
                state, reward, done, info = self.env.step(int(action))
                Reward += reward 
                j += 1 
            print("Test case: ", i,  ' Reward: ', Reward, "info: ", info)
            AvgReward += Reward/Nsamp
        self.env.close()
        print("Average over ", Nsamp, " samples = ", AvgReward)


class Agent(ptan.agent.PolicyAgent):

    @torch.no_grad()
    def __call__(self, states):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
                if states.ndim == 1:
                    states = states.view(1, -1)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = torch.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions)


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.n_actions = n_actions

    def forward(self, x):
        return self.net(x)
    
    def predict_probs(self, states):
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        logits = self.net(states).detach()
        probs = F.softmax(logits, dim = -1).numpy()
        return probs

    def get_action(self, states):
        probs = self.predict_probs(states)
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs

    def generate_session(self, env, t_max=1000):
        states, traj_probs, actions, rewards, dones = [], [], [], [], []
        s = env.reset()
        q_t = 1.0
        for t in range(t_max):
            a, action_probs = self.get_action(s)
            new_s, r, done, info = env.step(a)            
            q_t *= action_probs[a]
            
            states.append(s)
            traj_probs.append(q_t)
            actions.append(a)
            rewards.append(r)
            dones.append(done)

            s = deepcopy(new_s)
            if done:
                break
                
        return states, actions, rewards, dones


class RewardNet(nn.Module):
    def __init__(self, input_size):
        super(RewardNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def visualize_net(self, agent_net, Npoints=20):
        xmin = [-1, -1, -10, -10, 0]
        xmax = [1, 1, 10, 10, 9]

        S1 = np.linspace(xmin[0], xmax[0], Npoints)
        S2 = np.linspace(xmin[1], xmax[1], Npoints)

        Reward = np.zeros((Npoints, Npoints))
        for i in range(Npoints):
            for j in range(Npoints):
                state = np.array([S1[i], S2[j], 0, 0])
                action, _ = agent_net.get_action(state)
                x = torch.cat([float32_preprocessor(state), float32_preprocessor([action])]).view(1, -1)
                r = self.forward(x)
                Reward[i,j] = r

        # Plotting 
        X, Y = np.meshgrid(S1, S2)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Reward, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('surface');
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        ax.view_init(azim=90, elev=90)
        return fig






