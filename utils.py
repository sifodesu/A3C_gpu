import numpy as np
import logging, argparse
from scipy.signal import lfilter
from skimage.transform import resize

RES_F = 80

def parse():
    args = argparse.ArgumentParser()
    args.add_argument('--env', default='SuperMarioWorld-Snes', type=str, help='Name of the working environment')
    args.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    args.add_argument('--numWorkers', default=4, type=int, help='Number of training workers')
    args.add_argument('--load', default=False, action='store_true', help='Load last checkpoint')
    args.add_argument('--test', default=False, action='store_true', help= 'When True, the current trained model will be tested without any further training')
    args.add_argument('--update_frequency', default=50, type=int, help='Number of steps between each global gradient update')
    args.add_argument('--no_render', default=True, action='store_false', help='Don\t display the game being used for training')
    args.add_argument('--average_decay', default=0.99, type=float, help='Weight of old average for the new average estimate')
    args.add_argument('--hidden_gru_size', default=256, type=int, help='GRU hidden layer size (used for short long term memory)')
    args.add_argument('--no_batch_norm', default=True, action='store_false', help='Don\'t se batchnorm after each convolution')
    args.add_argument('--deeper', default=False, action='store_true', help='use deeper actor/critic')
    args.add_argument('--envState', default='Start.state', type=str, help='Name of the env state to load in the game')
    print( args.parse_args())
    return args.parse_args()
    


def discount_sum(x,discount):
    return lfilter([1],[1,-discount],x[::-1])[::-1] 

def preprocess_frame(frame):
    return resize(frame[35:195].mean(2), (RES_F,RES_F)).astype(np.float32).reshape(1,RES_F,RES_F)/255.

def sync_gradients(shared_model, local_model):
    local_params = local_model.parameters()
    for shared_param in shared_model.parameters():
        local_grad = next(local_params)._grad
        shared_param._grad = local_grad if not shared_param.grad else shared_param._grad


def record(msg):
    logging.info(msg)
    print(msg)
