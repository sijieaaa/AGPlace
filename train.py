
from tools import options as parser
opt = parser.parse_arguments()
from tools.options import logging_info, logging_init, get_datetime

import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
import time

import util
import test
import commons
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
from tools.options import logging_info, logging_init, logging_end

from datasets.datasets_ws_kitti360 import KITTI360BaseDataset 
from datasets.datasets_ws_kitti360 import KITTI360TripletsDataset
from datasets.datasets_ws_kitti360 import kitti360_collate_fn
from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_db
from datasets.datasets_ws_kitti360 import kitti360_collate_fn_cache_q

from datasets.datasets_ws_nuscenes import NuScenesBaseDataset
from datasets.datasets_ws_nuscenes import NuScenesTripletsDataset
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_db
from datasets.datasets_ws_nuscenes import nuscenes_collate_fn_cache_q



from network_mm.mm import MM    



from models_baseline.dbvanilla2d import DBVanilla2D
import torchvision.models as TVM

from compute_other_loss import compute_other_loss




def compute_loss(args, criterion_triplet, triplets_local_indexes, features):
    loss_triplet = 0
    
    if args.criterion == "triplet":
        triplets_local_indexes = torch.transpose(
            triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
        for triplets in triplets_local_indexes:
            queries_indexes, positives_indexes, negatives_indexes = triplets.T
            loss_triplet += criterion_triplet(features[queries_indexes],
                                            features[positives_indexes],
                                            features[negatives_indexes])
    elif args.criterion == 'sare_joint':
        # sare_joint needs to receive all the negatives at once
        triplet_index_batch = triplets_local_indexes.view(args.train_batch_size, 10, 3)
        for batch_triplet_index in triplet_index_batch:
            q = features[batch_triplet_index[0, 0]].unsqueeze(0)  # obtain query as tensor of shape 1xn_features
            p = features[batch_triplet_index[0, 1]].unsqueeze(0)  # obtain positive as tensor of shape 1xn_features
            n = features[batch_triplet_index[:, 2]]               # obtain negatives as tensor of shape 10xn_features
            loss_triplet += criterion_triplet(q, p, n)
    elif args.criterion == "sare_ind":
        for triplet in triplets_local_indexes:
            # triplet is a 1-D tensor with the 3 scalars indexes of the triplet
            q_i, p_i, n_i = triplet
            loss_triplet += criterion_triplet(features[q_i:q_i+1], features[p_i:p_i+1], features[n_i:n_i+1])
    
    del features
    loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

    return loss_triplet





def main():

    torch.backends.cudnn.benchmark = True  # Provides a speedup
    #### Initial setup: parser, logging...
    args = parser.parse_arguments()
    start_time = datetime.now()
    # args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    args.save_dir = join("logs", args.exp_name)
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")


    if opt.dataset == 'kitti360':
        triplets_ds = KITTI360TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = KITTI360BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    elif opt.dataset == 'nuscenes':
        triplets_ds = NuScenesTripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
        test_ds = NuScenesBaseDataset(args, args.datasets_folder, args.dataset_name, "test")

    logging.info(f"Train query set: {triplets_ds}")
    logging_info(f"Train query set: {triplets_ds}")
    logging.info(f"Test set: {test_ds}")
    logging_info(f"Test set: {test_ds}")




    #---- model db
    if args.modeldb == 'vanilla2d':
        model = DBVanilla2D(mode='db', dim=args.features_dim)

    #---- model q
    if args.modelq == 'mm':
        modelq = MM()


    else:
        raise NotImplementedError

    # print num of q parameters
    num_params_q = sum(p.numel() for p in modelq.parameters())
    logging.info(f"Number of parameters in modelq: {num_params_q}")
    logging_info(f"Number of parameters in modelq: {num_params_q}")

    # print num of db parameters
    num_params_db = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters in model: {num_params_db}")
    logging_info(f"Number of parameters in model: {num_params_db}")



    model = model.to(args.device)
    modelq = modelq.to(args.device)
    logging_info(f"Model: {model.__class__.__name__}")
    logging.info(f"Model: {model.__class__.__name__}")
    

    logging_info(f"Modelq: {modelq.__class__.__name__}")
    logging.info(f"Modelq: {modelq.__class__.__name__}")

    logging_info(f"Device: {args.device}")
    logging.info(f"Device: {args.device}")


    if args.aggregation in ["netvlad", "crn"]:  # If using NetVLAD layer, initialize it
        if not args.resume:
            triplets_ds.is_inference = True
            model.aggregation.initialize_netvlad_layer(args, triplets_ds, model.backbone)
            modelq.aggregation.initialize_netvlad_layer(args, triplets_ds, modelq.backbone)
        args.features_dim *= args.netvlad_clusters




    # ==== model db
    if isinstance(model, DBVanilla2D):
        params_db = []
        params_db.append({'params': model.parameters(), 'lr': args.lrdb})


    # ==== model query
    if isinstance(modelq, MM):
        params_q = []
        params_q.append({'params': modelq.image_fe.parameters(), 'lr': args.lr})
        params_q.append({'params': modelq.image_pool.parameters(), 'lr': args.lr})
        params_q.append({'params': modelq.vox_fe.parameters(), 'lr': args.lrpc})
        params_q.append({'params': modelq.vox_pool.parameters(), 'lr': args.lrpc})
        params_q.append({'params': modelq.fuseblocktoshallow.parameters(), 'lr': args.lr})
        params_q.append({'params': modelq.stg2fuseblock.parameters(), 'lr': args.lr})
        params_q.append({'params': modelq.stg2fusefc.parameters(), 'lr': args.lr})
        params_q.append({'params': modelq.image_weight, 'lr': args.lr})
        params_q.append({'params': modelq.vox_weight, 'lr': args.lrpc})
        params_q.append({'params': modelq.shallow_weight, 'lr': args.lr})
        params_q.append({'params': modelq.imageorg_weight, 'lr': args.lr})
        params_q.append({'params': modelq.voxorg_weight, 'lr': args.lr})
        params_q.append({'params': modelq.shalloworg_weight, 'lr': args.lr})
        params_q.append({'params': modelq.stg2image_weight, 'lr': args.lr})
        params_q.append({'params': modelq.stg2vox_weight, 'lr': args.lr})
        params_q.append({'params': modelq.stg2fuse_weight, 'lr': args.lr})
    

    if opt.share_qdb == True:
        model = modelq
        params_db = [{'params': torch.empty(0), 'lr': args.lrdb}]
        logging.info(f"Sharing weights... {modelq.__class__.__name__} and {model.__class__.__name__}")



    if args.aggregation == "crn":
        crn_params = list(model.module.aggregation.crn.parameters())
        net_params = list(model.module.backbone.parameters()) + \
            list([m[1] for m in model.module.aggregation.named_parameters() if not m[0].startswith('crn')])
        if args.optim == "adam":
            optimizer = torch.optim.Adam([{'params': crn_params, 'lr': args.lr_crn_layer},
                                        {'params': net_params, 'lr': args.lr_crn_net}])
            logging.info("You're using CRN with Adam, it is advised to use SGD")
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD([{'params': crn_params, 'lr': args.lr_crn_layer, 'momentum': 0.9, 'weight_decay': 0.001},
                                        {'params': net_params, 'lr': args.lr_crn_net, 'momentum': 0.9, 'weight_decay': 0.001}])
    else:
        if args.optim == "adam":
            optimizer = torch.optim.Adam(params_db)
            optimizerq = torch.optim.Adam(params_q)



    num_params_db = sum(pp.numel() for p in optimizer.param_groups for pp in p['params'])
    num_params_q = sum(pp.numel() for p in optimizerq.param_groups for pp in p['params'])
    logging.info(f"Number of parameters in optimizerdb: {num_params_db}")
    logging_info(f"Number of parameters in optimizerdb: {num_params_db}")
    logging.info(f"Number of parameters in optimizerq: {num_params_q}")
    logging_info(f"Number of parameters in optimizerq: {num_params_q}")


    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    elif args.criterion == "sare_ind":
        criterion_triplet = sare_ind
    elif args.criterion == "sare_joint":
        criterion_triplet = sare_joint


    #### Resume model, optimizer, and other training parameters
    if args.resume:
        if args.aggregation != 'crn':
            model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
        else:
            # CRN uses pretrained NetVLAD, then requires loading with strict=False and
            # does not load the optimizer from the checkpoint file.
            model, modelq, _, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, modelq, strict=False)
        logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
    else:
        best_r5 = start_epoch_num = not_improved_num = 0


    if args.backbone.startswith('vit'):
        logging.info(f"Output dimension of the model is {args.features_dim}")
    else:
        logging.info(f"Output dimension of the model is {args.features_dim}, with {util.get_flops(model, args.resize)}")


    if torch.cuda.device_count() >= 2:
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        model = convert_model(model)
        model = model.cuda()





    #### Training loop
    best_r1r5r10ep = [0, 0, 0, 0]
    for epoch_num in range(start_epoch_num, args.epochs_num):
        t0 = time.time()
        logging.info(f"Start training epoch: {epoch_num:02d}")
        
        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        
        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            # break
            logging.debug(f"Cache: {loop_num} / {loops_num}")
            
            # Compute triplets to use in the triplet loss
            triplets_ds.is_inference = True
            logging.info('compute triplets')
            triplets_ds.compute_triplets(args, model, modelq)
            triplets_ds.is_inference = False
            if opt.dataset == 'kitti360':
                triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                        batch_size=args.train_batch_size,
                                        collate_fn=kitti360_collate_fn,
                                        pin_memory=(args.device == "cuda"),
                                        drop_last=True)
            elif opt.dataset == 'nuscenes':
                triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                        batch_size=args.train_batch_size,
                                        collate_fn=nuscenes_collate_fn,
                                        pin_memory=(args.device == "cuda"),
                                        drop_last=True)
            
            model = model.train()
            modelq = modelq.train()

                    
            logging.info('start triplet training')

            for data_dict, triplets_local_indexes, _ in tqdm(triplets_dl):

                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)


                with torch.set_grad_enabled(args.train_modelq):
                    feats_ground = modelq(data_dict, mode='q') # [b,c]
                    feats_ground_embed = feats_ground['embedding']
                    feats_ground_embed = feats_ground_embed.unsqueeze(1) # [b,1,c]



                # dbs
                with torch.set_grad_enabled(args.train_modeldb):
                    feats_aerial = model(data_dict, mode='db') # [b,11,c]
                    feats_aerial_embed = feats_aerial['embedding']

                
                # break
                loss = 0
                if opt.modelq == 'mm':
                    otherloss = compute_other_loss(feats_ground, feats_aerial, data_dict, 
                                                positive_thd=opt.train_positives_dist_threshold,
                                                negative_thd=opt.val_positive_dist_threshold)
                    loss += otherloss



                # cat
                feats = torch.cat((feats_ground_embed, feats_aerial_embed), dim=1)
                feats = feats.view(-1, args.features_dim)
                triplet_loss = compute_loss(args, criterion_triplet, triplets_local_indexes, feats)
                loss += triplet_loss * opt.tripletloss_weight
                del feats

                optimizer.zero_grad()
                optimizerq.zero_grad()
                loss.backward()
                optimizer.step()
                optimizerq.step()
                
                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss
            
            
            logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                        f"current batch loss = {batch_loss:.4f}, " +
                        f"average epoch loss = {epoch_losses.mean():.4f}")
        



        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                    f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        

        recalls, _, _ = test.test(args, test_ds, model, modelq=modelq)


        is_best = sum(recalls[:3])>sum(best_r1r5r10ep[:3])
        if sum(recalls[:3])>sum(best_r1r5r10ep[:3]): # r@1 5 10
            best_r1r5r10ep[0] = recalls[0]
            best_r1r5r10ep[1] = recalls[1]
            best_r1r5r10ep[2] = recalls[2]
            best_r1r5r10ep[3] = epoch_num
        logging.info(f"Now : R@1 = {recalls[0]:.1f}   R@5 = {recalls[1]:.1f}   R@10 = {recalls[2]:.1f}   epoch = {epoch_num:d}")
        logging_info(f"Now : R@1 = {recalls[0]:.1f}   R@5 = {recalls[1]:.1f}   R@10 = {recalls[2]:.1f}   epoch = {epoch_num:d}")
        logging.info(f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")
        logging_info(f"Best: R@1 = {best_r1r5r10ep[0]:.1f}   R@5 = {best_r1r5r10ep[1]:.1f}   R@10 = {best_r1r5r10ep[2]:.1f}   epoch = {best_r1r5r10ep[3]:d}")


        
        # Save checkpoint, which contains all training parameters
        if epoch_num > 40:
            util.save_checkpoint(args, {
                "epoch_num": epoch_num, 
                'modelq_state_dict': modelq.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "recalls": recalls, 
                "best_r5": best_r5,
                "not_improved_num": not_improved_num
            }, is_best, filename=f"ep@{epoch_num}__r1@{recalls[0]:.0f}.pth")
        


        logging_info(f'{get_datetime()}')
        logging.info(f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')
        logging_info(f'---------------------------------- epoch: {epoch_num}   time: {time.time()-t0:.2f}')





    logging_end()


    #### Test best model on test set
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
    # logging.info(f"Recalls on {test_ds}: {recalls_str}")



if __name__ == "__main__":
    logging_init()
    main()
    logging_end()