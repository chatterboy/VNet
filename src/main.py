import os
import shutil

from src.network.vnet import VNet

config = dict()

# This can be done by a user want to use VNetx
config['epochs'] = 3000
config['batch_size'] = 2
config['learning_rate'] = 0.0001
config['momentum'] = 0.99
config['epoch_step'] = 5
config['valid_step'] = 1500
config['base_path'] = os.path.abspath('..')
config['train_path'] = 'data/train'
config['test_path'] = 'data/test'
config['chks_path'] = 'checkpoint'
config['flag'] = 'train' # train, retrain, test

if config['flag'] == 'train':
    chksPath = os.path.join(config['base_path'], 'checkpoint')
    if os.path.exists(chksPath):
        # TODO: how can i leave the info log efficiently without print()
        shutil.rmtree(chksPath)
        os.mkdir(chksPath)

vnet = VNet(config)

if config['flag'] == 'train':
    vnet.train()
elif config['flag'] == 'retrain':
    vnet.retrain()
elif config['flag'] == 'test':
    vnet.test()