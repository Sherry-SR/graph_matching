import importlib

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from operator import itemgetter
import pickle
from shutil import copyfile

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset

from nilearn.connectome import ConnectivityMeasure

from utils.helper import read_xlsx

class PNCEnrichedSet(InMemoryDataset):
    def __init__(self, sub_list, output, root, path_data, path_label, target_name = None, feature_mask = None, **kwargs):
        self.path = [path_data, path_label]
        self.output = output
        if sub_list is None:
            sub_list = os.listdir(path_data)
        self.sub_list = sub_list
        self.target_name = target_name
        self.feature_mask = feature_mask
        super(PNCEnrichedSet, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pnc_features_raw.pkl']

    @property
    def processed_file_names(self):
        return [self.output, '_'.join([self.output.split('.')[0], self.target_name, 'hist.npy'])]

    def download(self):
        assert len(self.path) == 2
        path_data = self.path[0]
        path_label = self.path[1]
        
        labels = read_xlsx(path_label)
        labels['Sex'], uniques = pd.factorize(labels['Sex'])
        labels = itemgetter('ScanAgeYears','Sex')(labels.set_index('Subject').to_dict())
        
        subjlist = [fname for fname in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, fname)) and fname != 'exclude']
        filelist = os.listdir(os.path.join(path_data, subjlist[0]))
        filelist.sort()
        ts_index = [i for i in range(len(filelist)) if 'timeseries' in filelist[i]]

        min_ts_length = None
        for subj in subjlist:
            print('checking', subj, '...')
            filename = filelist[ts_index[0]]
            filepath = os.path.join(path_data, subj, filename)
            if not os.path.exists(filepath):
                continue
            matrix = np.loadtxt(filepath)
            if min_ts_length is None or matrix.shape[0] < min_ts_length:
                min_ts_length = matrix.shape[0]
    
        with open(os.path.join(self.raw_dir, 'pnc_enriched_raw_info.txt'), 'w') as f:
            print('Label info:', file = f)
            print('All labels:', 'ScanAgeYears','Sex')
            print('Sex labels (0/1):', uniques.values, file = f)
            print('Timeseries length (min):', min_ts_length, file = f)
            print('\n', file = f)
            print('Features:', file = f)
            print(filelist, sep='\n', file = f)
            print('\n', file = f)
            print('Saved subjects:', file = f)
            print(subjlist, sep='\n', file = f)
            print('\n', file = f)

        with open(os.path.join(self.raw_dir, 'pnc_features_raw.pkl'), 'wb') as f:
            pickle.dump([labels, path_data, filelist, subjlist, min_ts_length], f)
            print('PNC dataset saved to path:', self.raw_dir)
        
    def process(self):
        with open(os.path.join(self.raw_dir, 'pnc_features_raw.pkl'), 'rb') as f:
            labels, path_data, filelist, _, min_ts_length = pickle.load(f)

        if self.feature_mask is not None:
            if np.isscalar(self.feature_mask):
                self.feature_mask = [i for i in range(len(filelist)) if self.feature_mask == int(filelist[i].split('_')[1])]
            filelist = [filelist[i] for i in self.feature_mask]
        ts_index = [i for i in range(len(filelist)) if 'timeseries' in filelist[i]]
        sc_index = [i for i in range(len(filelist)) if 'connmat' in filelist[i]]

        dataset_list = []
        sub_list = np.loadtxt(self.sub_list, dtype = str, delimiter = '\n')
        epsilon = 1e-5

        y = []

        for subj in sub_list:
            print('processing', subj, '...')
            features = []
            for filename in filelist:
                filepath = os.path.join(path_data, subj, filename)
                if not os.path.exists(filepath):
                    raise ValueError('invalid path '+filepath)
                matrix = np.loadtxt(filepath)
                features.append(matrix)

            data = Data(x = None, y = None)
            sub_labels = {'ScanAgeYears': labels[0][subj], 'Sex': labels[1][subj]}
            data.subj = int(subj.split('_')[0])
            if self.target_name is not None:
                data.y = sub_labels[self.target_name]
                y.append(data.y)
            data.labels = torch.tensor(list(sub_labels.values()))
            ts = []
            for i in ts_index:
                ts.append(features[i][:min_ts_length, :])
            data.fconn = torch.tensor(ConnectivityMeasure(kind='correlation').fit_transform(ts), dtype=torch.float32)
            sc = []
            sconn = []
            for i in sc_index:
                sc_matrix = features[i]
                sc.append(sc_matrix)
                D_mod = np.diag(np.sum(sc_matrix, dim=0))**(-1/2)
                sconn.append(D_mod @ sc_matrix @ D_mod)
            data.sconn = torch.tensor(sconn, dtype=torch.float32)
            dataset_list.append(data)

        self.data, self.slices = self.collate(dataset_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
        print('Processed dataset saved as', self.processed_paths[0])

        _, counts = np.unique(np.floor((np.array(y)-8)/2), return_counts = True)
        np.save(self.processed_paths[1], counts)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def get_data_loaders(config):
    assert 'loaders' in config, 'Could not find loaders configuration'
    loaders_config = config['loaders']
    class_name = loaders_config.pop('name')
    loader_class_name = loaders_config.pop('loader_name')
    train_list = loaders_config.pop('train_list')
    train_val_ratio = loaders_config.pop('train_val_ratio')
    test_list = loaders_config.pop('test_list')
    output_train = loaders_config.pop('output_train')
    output_test = loaders_config.pop('output_test')
    batch_size = loaders_config.pop('batch_size')

    m = importlib.import_module('utils.data_handler')
    clazz = getattr(m, class_name)
    m = importlib.import_module('torch_geometric.data')
    loader_clazz =getattr(m, loader_class_name)

    if train_list is None:
        train_val_dataset = None
    else:
        train_val_dataset = clazz(train_list, output_train, **loaders_config)#.shuffle()

    if train_val_ratio is not None:
        split_index = int(len(train_val_dataset) * train_val_ratio[0] / np.sum(train_val_ratio))
        train_dataset = train_val_dataset[:split_index]
        val_dataset = train_val_dataset[split_index:]
    else:
        train_dataset = train_val_dataset
        val_dataset = None
    
    if test_list is None:
        test_dataset = None
    else:
        test_dataset = clazz(test_list, output_test, **loaders_config)

    return {
        'train': loader_clazz(train_dataset, batch_size=batch_size, shuffle=True),
        'val': loader_clazz(val_dataset, batch_size=batch_size, shuffle=True),
        'test': loader_clazz(test_dataset, batch_size=batch_size, shuffle=True)
        }