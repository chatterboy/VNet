import os
import shutil

from src.network.vnet import VNet

config = dict()

# This can be done by a user want to use VNet
config['epochs'] = 10000
config['batch_size'] = 2
config['learning_rate'] = 0.0001
config['momentum'] = 0.99
config['epoch_step'] = 10
config['chks_step'] = 2500
config['base_path'] = 'C:/Users/AIserver/Documents/Github/VNet'
config['train_path'] = 'data/train'
config['test_path'] = 'data/test'
config['chks_path'] = 'checkpoint'

chksPath = os.path.join(config['base_path'], 'checkpoint')
if os.path.exists(chksPath):
    # TODO: how can i leave the info log efficiently without print()
    print('removing a directory: {}'.format(chksPath))
    print('all files are removing in {}'.format(chksPath))
    shutil.rmtree(chksPath)

print('making a new directory: {}'.format(chksPath))
os.mkdir(chksPath)

vnet = VNet(config)

# TODO: specify whether a user want to train this model
vnet.train()

# TODO: specify whether a user want to test this model
# vnet.test()