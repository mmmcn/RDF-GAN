"""
    Author: wmy
    Description: test models and datasets
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Simple test')
    parser.add_argument('--case', type=str, required=True, choices=['test_nyuv2s2d',
                                                                    'test_nyuv2r2r',
                                                                    'test_sunrgbd',
                                                                    'test_ddrnet_human'])

    args = parser.parse_args()

    return args


def test_ddrnet_human():
    from lib.dataset.ddrnet_human.ddrnet_human_dataset import DDRNetHumanDataset
    val_dataset = DDRNetHumanDataset(data_root='data/ddrnet_human', mode='test')

    print('len(val_dataset): {}'.format(len(val_dataset)))
    val_dataset.stat_depth()
    # val_dataset.get_test_data(0)

    train_dataset = DDRNetHumanDataset(data_root='data/ddrnet_human', mode='train')

    print('len(train_dataset): {}'.format(len(train_dataset)))
    train_dataset.stat_depth()





if __name__ == '__main__':

    args = parse_args()
    func = globals()[args.case]
    func()
