import os
import shutil

from src.network.vnet import VNet

config = dict()

# This can be done by a user want to use VNetx
config['epochs'] = 70000
config['batch_size'] = 2
config['learning_rate'] = 0.001
config['momentum'] = 0.9
config['epoch_step'] = None
config['valid_step'] = 5000
config['base_path'] = os.path.abspath('..')
config['temp_path'] = 'tmp'
config['result_path'] = 'result'
config['train_path'] = 'data/train'
config['train_data_path'] = os.path.join(config['base_path'], config['train_path'])
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