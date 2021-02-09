import torch.nn as nn
import glob
import numpy as np
import torch
import os.path
class A3C_Model(nn.Module): 
    def __init__(self, channels, hidden_gru_size, num_actions, use_batch_norm = True, deeper=False):
        super(A3C_Model, self).__init__()

        cnn_layers = []
        for num_features in (channels,32,32,32):
            cnn_layers.append(nn.Conv2d(num_features, 32, 3, stride=2, padding=1))
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(32))
            cnn_layers.append(nn.ReLU())
        cnn_layers.append(nn.Flatten())
        self.extract_features_cnn = nn.Sequential( *(layer for layer in cnn_layers) )

        self.gru = nn.GRUCell(32 * 5 * 5, hidden_gru_size)
        if deeper:
                self.actor = nn.Sequential(nn.Linear(hidden_gru_size, int(hidden_gru_size/2)),
				   	    nn.Tanh(),
				            nn.Linear(int(hidden_gru_size/2), num_actions))
                self.critic = nn.Sequential(nn.Linear(hidden_gru_size, int(hidden_gru_size/2)),
		                            nn.Tanh(),
		                            nn.Linear(int(hidden_gru_size/2), 1))
        else:
                self.actor = nn.Sequential(nn.Linear(hidden_gru_size, int(num_actions)))
                self.critic = nn.Sequential(nn.Linear(hidden_gru_size, 1))

    def forward(self, input):
        x, gru_x = input
        x = self.extract_features_cnn(x)
        gru_x = self.gru(x, gru_x)
        return self.actor(gru_x), self.critic(gru_x), gru_x

    def load_checkpoint(self, checkpoint_path):
        checkpoint_paths = glob.glob(os.path.join(checkpoint_path, '*.backup'))
        print(checkpoint_paths)
        print(os.path.join(checkpoint_path, '*.ckpt'))
        if checkpoint_paths:
            frames = [int(p.split('.')[-2]) for p in checkpoint_paths]
            eps = [int(p.split('.')[-3]) for p in checkpoint_paths]

            self.load_state_dict(torch.load(checkpoint_paths[np.argmax(frames)]))
            print(f'Checkpoint loaded: {checkpoint_paths[np.argmax(frames)]}')
      
            return (eps[np.argmax(frames)], max(frames))
        return None

#imported
class SharedOptim(torch.optim.Adam): 
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedOptim, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 
            super.step(closure)