import importlib

import os
import numpy as np
import pandas as pd

import pickle
import argparse
import logging

from utils.distance_requester import distance_requester

def get_pairs(subj_list, label_list):
    patient = subj_list[label_list == 'p']
    control = subj_list[label_list == 'c']
    pairs = []
    for subj_p in patient:
        for subj_c in control:
            if subj_p == 'p021' or subj_p == 'p009':
                continue
            pairs.append((subj_p, subj_c))
    for i in range(len(control)):
        for j in range(i+1, len(control)):
            pairs.append((control[i], control[j]))    
    return pairs

def compute_pairwise_dist(theta, data):
    pairs, conn_dict, df_roi, networks = data
    node_factor = np.empty(len(df_roi))
    for i in range(len(networks)):
        node_factor[df_roi['Network'] == networks[i]] = theta[i]
    node_factor = [node_factor, node_factor]
    
    dist_dic = {}
    dist_pc = {}
    dist_cc = []
    for pair in pairs:
        (s1, s2) = pair
        conn_s1, conn_s2 = conn_dict[s1]['s1'], conn_dict[s2]['s1']
        dist_dic[pair] = distance_requester(conn_s1, conn_s2).graphedit(node_factor)
        dd = (dist_dic[pair][1] == np.arange(len(dist_dic[pair][1]))).mean()
        if s1[0] == 'p':
            if dist_pc.get(s1) is None:
                dist_pc[s1] = []
            dist_pc[s1].append(dd)
        else:
            dist_cc.append(dd)

    dist_cc_mean = np.mean(dist_cc)
    distdiff = [np.array(x).mean()-dist_cc_mean for x in dist_pc.values()]

    subjects = list(dist_dic.keys())
    dx_groups = [x[0][0]+x[1][0] for x in subjects]
    graphedit = [x[0] for x in dist_dic.values()]
    graphedit_match = [x[1] for x in dist_dic.values()]
    df_dist = pd.DataFrame({'subjects': subjects,
                       'dx_groups':dx_groups,
                       'times': ['s1s1']*len(subjects),
                       'graphedit': graphedit,
                       'graphedit_match': graphedit_match})

    return distdiff, df_dist

def prior(theta):
    if np.all(theta > 0):
        return 1
    return 0

def posterior(theta, data, gamma = 0.5):
    prior_theta = prior(theta)
    if prior_theta == 0:
        return float('-inf'), None
    distdiff, df_dist = compute_pairwise_dist(theta, data)
    energy = [max(0, x + gamma) for x in distdiff]
    return - np.sum(energy) + np.log(prior_theta), df_dist

def transition(theta):
    theta_new = np.random.multivariate_normal(theta, 0.001 * np.eye(len(theta)))
    theta_new[theta_new<=1e-6] = 1e-6
    theta_new = theta_new / theta_new.sum()
    return theta_new

def acceptance(p, p_new, iter, Tc = 0.01):
    T = Tc / np.log(iter + 1)
    if p_new > p:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < (np.exp((p_new - p) / T)))

def metropolis_hastings(posterior_computer, transition_model, param_init, iterations, data, acceptance_rule, logger, out_path, start_iter = 0):
    logger.info('metropolis hasting...')
    if not os.path.exists(os.path.join(out_path, 'mcmc_output')):
        os.makedirs(os.path.join(out_path, 'mcmc_output'))
    if start_iter == 0:
        #param_init = param_init / np.sum(param_init)
        p_new, df_dist = posterior_computer(param_init, data)
        param_list = [param_init]
        p_list = [p_new]
        accepted = [0]
        rejected = []
        with open(os.path.join(out_path, 'mcmc_output', 'iter0.pkl'), 'wb') as f:
            pickle.dump([param_list, p_list, accepted, rejected, df_dist], f)
        logger.info('iter 0/%d finished!' % (iterations))
        start_iter = start_iter + 1
    else:
        with open(os.path.join(out_path, 'mcmc_output', 'iter'+str(start_iter-1)+'.pkl'), 'rb') as f:
            param_list, p_list, accepted, rejected, _ = pickle.load(f)
        logger.info('continuing from iter %d/%d...' % (start_iter-1, iterations))
    for i in range(start_iter, iterations):
        theta_new =  transition_model(param_list[accepted[-1]])
        p_new, df_dist = posterior_computer(theta_new, data)
        param_list.append(theta_new)
        p_list.append(p_new)
        if (acceptance_rule(p_list[accepted[-1]], p_new, i)):
            accepted.append(i)
        else:
            rejected.append(i)
        with open(os.path.join(out_path, 'mcmc_output', 'iter'+str(i)+'.pkl'), 'wb') as f:
            pickle.dump([param_list, p_list, accepted, rejected, df_dist], f)
        logger.info('iter %d/%d finished!' % (i, iterations))
    return param_list, p_list, accepted, rejected

def main():
    parser = argparse.ArgumentParser(description='train mcmc for graph edit distance')
    parser.add_argument('-d', '--data_dir', type=str, default='../Data/TBI/TBI_Connectomes_wSubcort', help='path to data')
    parser.add_argument('-a', '--atlas', type=str, default='../Data/TBI/atlas/Schaefer2018_116Parcels_7Networks_LookupTable.csv', help='path to atlas')
    parser.add_argument('-o', '--output_dir', type=str, default='../Results/tbi_mcmc_exp01', help='path to outputs')
    parser.add_argument('-n', '--n_node', type=str, default='116', help='number of node to use for rois')
    parser.add_argument('-m', '--mode', type=str, default='DTI_det', help='mode of connectivity, DTI_det, DTI_prob, or Restbold')
    parser.add_argument('-l', '--list', type=str, default='../Results/tbi_mcmc_exp01/cv_list', help='path to train/test list')
    parser.add_argument('-f', '--fold', type=str, default='0', help='fold')
    args = parser.parse_args()

    fold_path = os.path.join(args.output_dir, 'fold'+args.fold)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)

    logger = logging.getLogger('Graph edit distance')
    hdlr = logging.FileHandler(os.path.join(fold_path, 'dist_ged.log'), 'w+')
    formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    df_roi = pd.read_csv(args.atlas, names = ['Index', 'Label'])
    df_roi = df_roi.drop(columns = ['Index'])
    df_roi['Network'] = [x.split('_')[2] if x.startswith('7Networks') else 'Sub' for x in df_roi['Label']]

    subj_list = [x for x in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, x))]
    subj_list.remove('p021')

    conn_dict = {}
    for subj in subj_list:
        filelist = [x for x in os.listdir(os.path.join(args.data_dir, subj))
                            if str(args.n_node) in x.split('_') and x.startswith(args.mode)]
        filelist.sort()
        conn_dict[subj] = {}
        for filename in filelist:
            filepath = os.path.join(args.data_dir, subj, filename)
            seq = filename.split('.')[0].split('_')[-1]
            conn_dict[subj][seq] = np.loadtxt(filepath)
    logger.info('finished loading..')

    train_list = np.loadtxt(os.path.join(args.list, 'train_list_fold'+args.fold+'.txt'), dtype=str)
    train_label = np.array([x[0] for x in train_list])
    train_pairs = get_pairs(train_list, train_label)

    test_list = np.loadtxt(os.path.join(args.list, 'test_list_fold'+args.fold+'.txt'), dtype=str)
    test_label = np.array([x[0] for x in test_list])
    test_pairs = get_pairs(test_list, test_label)

    networks = np.unique(df_roi['Network'])

    logger.info('Start training fold'+args.fold+'...')
    param_init = np.ones(len(networks))
    iterations = 100
    start_iter = 0
    data = [train_pairs, conn_dict, df_roi, networks]
    param_list, p_list, accepted, rejected = metropolis_hastings(posterior, transition,
                                                                 param_init, iterations,
                                                                 data, acceptance,
                                                                 logger, fold_path, start_iter)
    logger.info('Starting testing fold'+args.fold+'...')
    with open(os.path.join(fold_path, 'mcmc_output', 'iter'+str(iterations-1)+'.pkl'), 'rb') as f:
        param_list, _, accepted, rejected, _ = pickle.load(f)

    p_list = []
    data = [test_pairs, conn_dict, df_roi, networks]

    for i in range(start_iter, iterations):
        theta_new =  param_list[i]
        p_new, df_dist = posterior(theta_new, data)

        p_list.append(p_new)
        with open(os.path.join(fold_path, 'mcmc_output', 'test_iter'+str(i)+'.pkl'), 'wb') as f:
            pickle.dump([param_list[:i+1], p_list, accepted[:i+1], rejected[:i+1], df_dist], f)
        logger.info('iter %d/%d finished!' % (i, iterations))

    train_list = train_list[train_label == 'c']
    test_pairs = []
    for s_test in test_list:
        if s_test == 'p021' or s_test == 'p009':
            continue
        for s_control in train_list:
            test_pairs.append((s_test, s_control))
    data = [test_pairs, conn_dict, df_roi, networks]
    p_new_start, df_dist_start = posterior(param_list[start_iter], data)
    p_new_last, df_dist_last = posterior(param_list[iterations-1], data)

    with open(os.path.join(fold_path, 'mcmc_output', 'testing.pkl'), 'wb') as f:
        pickle.dump([param_list[start_iter], p_new_start, df_dist_start, param_list[iterations-1], p_new_last, df_dist_last], f)

    logger.info('testing finished!')

if __name__ == "__main__":
    main()