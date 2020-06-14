import importlib

import os
import numpy as np
import pandas as pd

import pickle
import argparse
import logging

from utils.distance_requester import distance_requester

def get_triplets(subj_list, label_list):
    patient = subj_list[label_list == 'p']
    control = subj_list[label_list == 'c']
    triplets = []
    for subj_p in patient:
        for i in range(len(control)):
            for j in range(i+1, len(control)):
                if subj_p == 'p021' or subj_p == 'p009':
                    continue
                triplets.append((subj_p, control[i], control[j]))
    return triplets

def compute_pairwise_dist(theta, data):
    triplets, conn_dict, df_roi, networks = data
    node_factor = np.empty(len(df_roi))
    for i in range(len(networks)):
        node_factor[df_roi['Network'] == networks[i]] = theta[i]
    node_factor = [node_factor, node_factor]
    dist_dic = {}
    for tri in triplets:
        (p, c1, c2) = tri
        conn_p, conn_c1, conn_c2 = conn_dict[p]['s1'], conn_dict[c1]['s1'], conn_dict[c2]['s1']
        dist_dic[(p, c1)] = distance_requester(conn_p, conn_c1).graphedit(node_factor)
        dist_dic[(p, c2)] = distance_requester(conn_p, conn_c2).graphedit(node_factor)
        if dist_dic.get((c1, c2)) is None:
            dist_dic[(c1, c2)] = distance_requester(conn_c1, conn_c2).graphedit(node_factor)
    distmax = max(dist_dic.values())
    dist_dic = {k: v/distmax for k,v in dist_dic.items()}

    distdiff = {}
    for tri in triplets:
        (p, c1, c2) = tri
        if distdiff.get(p) is None:
            distdiff[p] = []
        else:
            distdiff[p].append(dist_dic[(c1, c2)][0] - dist_dic[p, c1][0])
            distdiff[p].append(dist_dic[(c1, c2)][0] - dist_dic[p, c2][0])
    
    subjects = list(dist_dic.keys())
    dx_groups = [x[0][0]+x[1][0] for x in subjects]
    graphedit = [x[0] for x in dist_dic.values()]
    graphedit_match = [x[1] for x in dist_dic.values()]
    distdiff = [np.array(x).mean() for x in distdiff.values()]
    df_dist = pd.DataFrame({'subjects': subjects,
                       'dx_groups':dx_groups,
                       'times': ['s1s1']*len(subjects),
                       'graphedit': graphedit,
                       'graphedit_match': graphedit_match})

    return distdiff, df_dist

def prior(theta):
    if np.all(theta >= 0):
        return 1
    return 0

def posterior(theta, data, iter, gamma = 0.5, Tc = 100):
    prior_theta = prior(theta)
    if prior_theta == 0:
        return float('-inf')
    distdiff, df_dist = compute_pairwise_dist(theta, data)
    energy = [max(0, x + gamma) for x in distdiff]
    T = Tc / np.log(iter + 2)
    return - np.sum(energy) / T + np.log(prior_theta), df_dist

def transition(theta):
    theta_new = np.random.multivariate_normal(theta, 0.01 * np.eye(len(theta)))
    theta_new[theta_new<0] = 0
    theta_new = theta_new / np.sum(theta_new.sum)
    return theta_new

def acceptance(p, p_new):
    if p_new > p:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < (np.exp(p_new - p)))

def metropolis_hastings(posterior_computer, transition_model, param_init, iterations, data, acceptance_rule, logger, out_path, start_iter = 0):
    logger.info('metropolis hasting...')
    if not os.path.exists(os.path.join(out_path, 'mcmc_output')):
        os.makedirs(os.path.join(out_path, 'mcmc_output'))
    if start_iter == 0:
        param_init = param_init / np.sum(param_init)
        p_new, df_dist = posterior_computer(param_init, data, 0)
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
        p_new, df_dist = posterior_computer(theta_new, data, i)
        param_list.append(theta_new)
        p_list.append(p_new)
        if (acceptance_rule(p_list[accepted[-1]], p_new)):
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
    parser.add_argument('-o', '--output_dir', type=str, default='../Results/tbi_mcmc_exp05', help='path to outputs')
    parser.add_argument('-n', '--n_node', type=str, default='116', help='number of node to use for rois')
    parser.add_argument('-m', '--mode', type=str, default='DTI_det', help='mode of connectivity, DTI_det, DTI_prob, or Restbold')
    parser.add_argument('-l', '--list', type=str, default='../Results/tbi_mcmc_exp01/cv_list', help='path to train/test list')
    parser.add_argument('-f', '--fold', type=str, default='0', help='fold')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = logging.getLogger('Graph edit distance')
    hdlr = logging.FileHandler(os.path.join(args.output_dir, 'dist_ged.log'), 'w+')
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
    train_triplets = get_triplets(train_list, train_label)

    test_list = np.loadtxt(os.path.join(args.list, 'test_list_fold'+args.fold+'.txt'), dtype=str)
    test_label = np.array([x[0] for x in test_list])
    test_triplets = get_triplets(test_list, test_label)

    networks = np.unique(df_roi['Network'])

    logger.info('Start training fold'+args.fold+'...')
    param_init = np.ones(len(networks))
    iterations = 100
    start_iter = 0
    data = [train_triplets, conn_dict, df_roi, networks]
    param_list, p_list, accepted, _ = metropolis_hastings(posterior, transition,
                                                                 param_init, iterations,
                                                                 data, acceptance,
                                                                 logger, args.output_dir, start_iter)
    logger.info('accepted p: ' + ' '.join(np.array(p_list)[accepted]))

    data = [test_triplets, conn_dict, df_roi, networks]
    p_test, df_dist_test = posterior(param_list[accepted[-1]], data, 0)
    with open(os.path.join(args.output_dir, 'mcmc_output', 'testing.pkl'), 'wb') as f:
        pickle.dump([param_list[accepted[-1]], p_test, df_dist_test], f)
    logger.info('testing finished!')
if __name__ == "__main__":
    main()