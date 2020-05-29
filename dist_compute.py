import importlib

import os
import numpy as np
import pickle
import argparse
import logging
import pandas as pd
import networkx as nx
from utils.distance_requester import distance_requester
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='compute distance for graphs')
parser.add_argument('-d', '--data_dir', type=str, default='../Data/TBI/TBI_Connectomes_wSubcort', help='path to data')
parser.add_argument('-o', '--output_dir', type=str, default='../Results/TBI', help='path to outputs')
parser.add_argument('-n', '--n_node', type=str, default=116, help='number of node to use for rois')
parser.add_argument('-m', '--mode', type=str, default='DTI_det', help='mode of connectivity, DTI_det, DTI_prob, or Restbold')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger = logging.getLogger('Compute Distance')
hdlr = logging.FileHandler(os.path.join(args.output_dir, 'dist_compute.log'))
formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

subj_list = [x for x in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, x))]
subj_list.remove('p021')
label_list = [x[0] for x in subj_list]
df = pd.DataFrame({'Subject':subj_list, 'DX_Group':label_list, 'Times':[[]]*len(subj_list)})
df = df.set_index('Subject')

conn_dict = {}
for subj in subj_list:
    filelist = [x for x in os.listdir(os.path.join(args.data_dir, subj))
                        if str(args.n_node) in x.split('_') and x.startswith(args.mode)]
    filelist.sort()
    conn_dict[subj] = {}
    for filename in filelist:
        filepath = os.path.join(args.data_dir, subj, filename)
        seq = filename.split('.')[0].split('_')[-1]
        df['Times'].loc[subj] = df['Times'].loc[subj] + [seq]
        conn_dict[subj][seq] = np.loadtxt(filepath)
logger.info('finished loading..')

dist_measures = ['graphedit', 'euclidean', 'canberra', 'pearson', 'spearman']
columns = ['subjects', 'dx_groups', 'times'] + dist_measures + ['pearson_p', 'spearman_p']

rows_list = {x:[] for x in columns}
for i in range(len(subj_list)):
    for j in range(i+1, len(subj_list)):
        isubj = subj_list[i]
        jsubj = subj_list[j]
        logger.info('computing: [%d] %s, [%d] %s'%(i, isubj, j, jsubj))
        flags = [df['DX_Group'].loc[isubj], df['DX_Group'].loc[jsubj]]
        flags.sort()
        flags = ''.join(flags)
        subjects = [isubj, jsubj]
        subjects.sort()
        for iseq in df['Times'].loc[isubj]:
            for jseq in df['Times'].loc[jsubj]:
                seqs = [iseq, jseq]
                seqs.sort()
                seqs = ''.join(seqs)
                iconn = conn_dict[isubj][iseq]
                jconn = conn_dict[jsubj][jseq]
                dist_req = distance_requester(iconn, jconn)
                rows_list['subjects'].append(subjects)
                rows_list['dx_groups'].append(flags)
                rows_list['times'].append(seqs)
                for dist_measure in dist_measures:
                    dist_method = getattr(dist_req, dist_measure)
                    dist = dist_method()
                    if isinstance(dist, tuple):
                        dist_measure_ext = [x for x in columns if x.startswith(dist_measure+'_')]
                        dist_measure_ext = dist_measure_ext[0]
                        rows_list[dist_measure_ext].append(dist[1:])
                        dist = dist[0]
                    rows_list[dist_measure].append(dist)

psubj_list = df[df['DX_Group'] == 'p'].index.tolist()
flags = 'p'
for isubj in psubj_list:
    seq_list = df['Times'].loc[isubj]
    logger.info('computing: [%d] %s'%(i, isubj))
    for i in range(len(seq_list)):
        for j in range(i+1, len(seq_list)):
            seqs = [seq_list[i], seq_list[j]]
            seqs.sort()
            seqs = ''.join(seqs)
            iconn = conn_dict[isubj][seq_list[i]]
            jconn = conn_dict[isubj][seq_list[j]]
            dist_req = distance_requester(iconn, jconn)
            rows_list['subjects'].append([isubj])
            rows_list['dx_groups'].append(flags)
            rows_list['times'].append(seqs)
            for dist_measure in dist_measures:
                dist_method = getattr(dist_req, dist_measure)
                dist = dist_method()
                if isinstance(dist, tuple):
                    dist_measure_ext = [x for x in columns if x.startswith(dist_measure+'_')]
                    dist_measure_ext = dist_measure_ext[0]
                    rows_list[dist_measure_ext].append(dist[1:])
                    dist = dist[0]
                rows_list[dist_measure].append(dist)

logger.info('saving...')
df_dist = pd.DataFrame(rows_list)

with open(os.path.join(args.output_dir, 'df_dist.pkl'), 'wb') as f:
    pickle.dump([df, conn_dict, df_dist], f)