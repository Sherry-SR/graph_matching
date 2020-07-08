import sys
import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='get distance matrix')
    parser.add_argument('-p', '--result_dir', type=str,
                        default='/home/sherry/Dropbox/PhD/Results/graph_matching',
                        help='output result path')
    parser.add_argument('-d', '--data_dir', type=str,
                        default='/home/sherry/Dropbox/PhD/Data/PNC_Enriched/PNC_Connectomes'
                        help='connectome data path')
    parser.add_argument('-s', '--subj_list', type=str,
                        default='/home/sherry/Dropbox/PhD/Data/PNC_Enriched/PNC_Connectomes/IntersectionList.txt'
                        help='subject list path')
    parser.add_argument('-t', '--trim_method', type=str,
                        default='demo',
                        help='truncated (min) or trim or full')
    parser.add_argument('-r', '--roi', type=int, 
                        default=100,
                        help='number of ROIs, default: 100')
    parser.add_argument('-m', '--max_workers', type=int, 
                        default=20,
                        help='number of concurrent threads, default: 20')
    args = parser.parse_args()

    # load analyzer class
    UTILS_DIR = './utils/FC_analyzer'
    sys.path.insert(0, UTILS_DIR)
    from whole_brain_FC_analyzer import distance_matrix_requestor

    dr = distance_matrix_requestor(args.data_dir, args.subj_list
                                   DIR=args.result_dir,
                                   trim_method=args.trim_method,
                                   max_workers=args.max_workers,
                                   kROI=args.roi)
    then = time.time()
    dr.make_distance_requests()
    print("elapsed time = %s seconds" %(time.time() - then))