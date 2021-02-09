import torch, logging, gym, os
import torch.multiprocessing as mp
from datetime import datetime
from a3c_model import *
from utils import *
from training import *

os.environ['LANG']='en_US'
os.environ['OMP_NUM_THREADS'] = '1'

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                if torch.cuda.is_available()
                                else torch.FloatTensor)
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    args = parse()
    args.checkpoint_path = args.env + '-hidden' + str(args.hidden_gru_size) + '-batch' + str(args.no_batch_norm) + ('.deeper' if args.deeper else '') + '/' 
    if args.test:
          args.numWorkers = 1 
          args.lr = 0
    args.num_actions = gym.make(args.env).action_space.n 
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path) 

    shared_model = A3C_Model(channels=1, hidden_gru_size=args.hidden_gru_size, num_actions=args.num_actions, use_batch_norm = args.no_batch_norm, deeper=args.deeper).share_memory()
    shared_optim = SharedOptim(shared_model.parameters(), lr=args.lr)

    log = dict()
    for attr in ['n_episode', 'n_frame', 'ep_loss_avg']:
        log[attr] = torch.DoubleTensor([0]).share_memory_()
    log['ep_reward_avg'] = torch.DoubleTensor([0]).share_memory_()
    
    if args.load:
        res = shared_model.load_checkpoint(args.checkpoint_path)
        if res != None:
            log['n_frame'] += res[1] * NB_FRAMES_CHECKPOINT
            log['n_episode'] += res[0]
    args.log_path = os.path.join(args.checkpoint_path, 'log.txt')
    if int(log['n_frame'].item()) == 0 and os.path.isfile(args.log_path):
        os.remove(args.log_path)
    print(f'Starting time: {datetime.now()}\n')

    workers = []
    mp.set_start_method('spawn')

    for worker_index in range(args.numWorkers):
        job = mp.Process(target=run, args=(args, log, shared_model, shared_optim, worker_index))
        job.start() 
        workers.append(job)
    for w in workers: 
        w.join()