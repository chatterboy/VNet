import os
import shutil

from src.network.vnet import VNet

config = dict()

# This can be done by a user want to use VNetx
config['epochs'] = 100000
config['batch_size'] = 2
config['learning_rate'] = 0.0001
config['momentum'] = 0.99
config['epoch_step'] = None
config['valid_step'] = 5000
config['base_path'] = os.path.abspath('..')
config['train_path'] = 'data/train'
config['test_path'] = 'data/test'
config['chks_path'] = 'checkpoint'
config['flag'] = 'train' # train, retrain, test

if config['flag'] == 'train':
    chksPath = os.path.join(config['base_path'], 'checkpoint')
    if os.path.exists(chksPath):
        shutil.rmtree(chksPath)
        os.mkdir(chksPath)

vnet = VNet(config)

if config['flag'] == 'train':
    vnet.train()
elif config['flag'] == 'retrain':
    vnet.retrain()
elif config['flag'] == 'test':
    vnet.test()