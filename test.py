
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from datasets_ws import collate_fn_cache_db
from datasets_ws import collate_fn_cache_q

from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_db
from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_q

from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_db
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_q

from tools.options import parse_arguments
opt = parse_arguments()




def compute_recall(args, queries_features, database_features, test_ds, test_method='hard_resize'):

    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    # del database_features, queries_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (test_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (test_ds.queries_num, 20 * 5))
        for q in range(test_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (test_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (test_ds.queries_num, 5, 20))
        for q in range(test_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    #### For each query, check if the predictions are correct
    positives_per_query = test_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / test_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str




def test(args, test_ds, model, test_method="hard_resize", pca=None, modelq=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                           "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    # if args.efficient_ram_testing:
    #     return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    model = model.eval()
    modelq = modelq.eval()



    
    with torch.no_grad():
        # ============ database
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        test_ds.test_method = "hard_resize"
        database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))

        if opt.dataset == 'kitti360':
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"),
                                            collate_fn=kitti360_collate_fn_cache_db)
        elif opt.dataset == 'nuscenes':
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"),
                                            collate_fn=nuscenes_collate_fn_cache_db)
            
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * test_ds.queries_num + test_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(test_ds), args.features_dim), dtype="float32")
        db_locations = []
        for data_dict, indices in tqdm(database_dataloader, ncols=50):
            for _k, _v in data_dict.items():
                if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)
            features = model(data_dict, mode='db')
            features = features['embedding']
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
            if opt.dataset == 'nuscenes':
                db_locations.extend(data_dict['db_location'])
        
        

        # ============ queries
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        test_ds.test_method = test_method
        queries_subset_ds = Subset(test_ds, list(range(test_ds.database_num, test_ds.database_num+test_ds.queries_num)))
        if opt.dataset == 'kitti360':
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"),
                                            collate_fn=kitti360_collate_fn_cache_q)
        elif opt.dataset == 'nuscenes':
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"),
                                            collate_fn=nuscenes_collate_fn_cache_q)
            
        q_locations = []
        for data_dict, indices in tqdm(queries_dataloader, ncols=50):
            for _k, _v in data_dict.items():
                if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)
            # images = data_dict['image']
            # hard_resize
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                images = torch.cat(tuple(images))  # shape = 5*bs x 3 x 480 x 480
            features = modelq(data_dict,mode='q')
            features = features['embedding']
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = test_ds.database_num + (indices[0] - test_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
            if opt.dataset == 'nuscenes':
                q_locations.extend(data_dict['query_location'])


    queries_features = all_features[test_ds.database_num:]
    database_features = all_features[:test_ds.database_num]
    if opt.dataset == 'nuscenes':
        assert len(q_locations) == len(queries_features)
        assert len(db_locations) == len(database_features)




    recalls, recalls_str = compute_recall(args, queries_features, database_features, test_ds, test_method)
    return recalls, recalls_str, None




def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n



if __name__ == '__main__':

    from datasets.datasets_ws_nuscenes import NuScenesBaseDataset
    from datasets.datasets_ws_nuscenes import NuScenesTripletsDataset

    from datasets.datasets_ws_kitti360 import KITTI360BaseDataset 
    from datasets.datasets_ws_kitti360 import KITTI360TripletsDataset
    from datasets.datasets_ws_kitti360 import kitti360_collate_fn
    from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_db
    from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_q


    # from network.dinov2_locnet import DinoV2LocNet
    # from network.image_locnet import ImageLocNet
    # from network.mm import MM
    # from network.mm3d import MM3D
    from network_mm.mm import MM
    # from network_mm.mmfc import MMFC

    # from models_baseline.minkloc3dv2 import MinkLoc3DV2
    # from models_baseline.bevplace import BEVPlace
    from models_baseline.minklocpp import MinkLocPP
    from models_baseline.adafusion import AdaFusion
    # from models_baseline.minklocppbev import MinkLocPPBEV
    # from models_baseline.adafusionbev import AdaFusionBEV
    # from models_baseline.minklocppsph import MinkLocPPSph
    # from models_baseline.minkloc3dbev import MinkLoc3DBEV
    # from models_baseline.lcpr import LCPR
    from models_baseline.UMF.umf import UMF
    from models_baseline.lcprours import LCPROurs
    from models_baseline.mssplace import MSSPlace
    # from models_baseline.vsgp import VSGP
    from models_baseline.dbvanilla2d import DBVanilla2D

    import torch

    from tools import options as parser
    args = parser.parse_arguments()


    if args.dataset == 'kitti360':
        test_ds = KITTI360BaseDataset(args, args.datasets_folder, args.dataset_name, "test")

    elif args.dataset == 'nuscenes':
        test_ds = NuScenesBaseDataset(args, args.datasets_folder, args.dataset_name, "test")

    args.features_dim = 256
    model = DBVanilla2D(mode='db', dim=args.features_dim)


    modelq = MM(drop=None) # need change opt
    print('Modeldb:', model.__class__.__name__)
    print(f'Modelq: {modelq.__class__.__name__}')


    pth = torch.load(


    )

    print('Loading modeldb weights...')
    print('Loading modelq weights...')
    model.load_state_dict(pth['model_state_dict'])
    modelq.load_state_dict(pth['modelq_state_dict'])

    model = model.to(args.device)
    modelq = modelq.to(args.device)

    print(f'Drop = {modelq.drop}...')
    recalls, recalls_str, recalls_dict = test(args, test_ds, model, modelq=modelq, test_method="hard_resize")
    print(recalls_str)
