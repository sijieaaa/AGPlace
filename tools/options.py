
import os
import torch
import argparse
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import MinkowskiEngine as ME
os.environ["OMP_NUM_THREADS"] = '8'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--machine", type=str, default='sijiewan')
    parser.add_argument("--dataset", type=str, default='kitti360') # kitti360  nuscenes  
    parser.add_argument("--datasets_folder", type=str, 
                        default=''
                        )
    parser.add_argument("--dataset_name", type=str, 
                        default=''
                        )
    parser.add_argument("--dataroot", type=str, 
                        )
    parser.add_argument("--maptype", type=str, default='satellite') # satellite  roadmap  terrain  hybrid
    parser.add_argument("--traindownsample", type=int, default=4) # 4
    parser.add_argument("--train_ratio", type=float, default=0.85) 
    # nuscenes: fl_f_fr_bl_b_br
    # kitti360: 00  0203 
    parser.add_argument("--camnames", type=str, default='00') 
    
    parser.add_argument("--train_batch_size", type=int, default=16, # 16
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=32, # 32
                        help="Batch size for inference (caching and testing)")
    
    parser.add_argument("--cache_refresh_rate", type=int, default=4000, # 1000, < len(train_dataset)
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=16000, # 5000
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25)
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10)
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")



    
    parser.add_argument("--epochs_num", type=int, default=100,
                        help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lrpc", type=float, default=1e-4)
    parser.add_argument("--lrdb", type=float, default=1e-5)
    parser.add_argument('--resize', type=int, default=[256,256], nargs=2, help="Resizing shape for images (HxW).") # database transform
    parser.add_argument('--color_jitter', type=float, default=0) # query transform
    parser.add_argument('--quant_size', type=float, default=2) # query transform

    parser.add_argument("--db_cropsize", type=int, default=256) # 256 384 512 640
    parser.add_argument("--db_resize", type=int, default=256)
    parser.add_argument("--db_jitter", type=float, default=0)
    parser.add_argument("--q_resize", type=int, default=256)
    parser.add_argument("--q_jitter", type=float, default=0)
    parser.add_argument("--sph_size", type=int, default=32) # [61,361] -> short-32
    

    parser.add_argument('--sph_jit', type=float, default=0.2)
    parser.add_argument('--bev_jit', type=float, default=0.2)

    
    parser.add_argument("--train_modeldb", type=str, default=True)
    parser.add_argument("--train_modelq", type=str, default=True)
    parser.add_argument("--share_db", type=str, default=False)
    parser.add_argument("--share_dbfe", type=str, default=False)
    parser.add_argument("--share_qdb", type=str, default=False)



    # mm  mmfc  minklocpp  adafusion
    parser.add_argument("--modelq", type=str, default='mm') #
    parser.add_argument("--features_dim", type=int, default=256) 
    parser.add_argument("--read_pc", type=str, default=True) 

    parser.add_argument("--modeldb", type=str, default='vanilla2d')

    # ==== for database
    parser.add_argument("--dbimage_fe", type=str, default='resnet18') # under vanilla2d
    parser.add_argument("--dbimage_fe_layers", type=str, default='2_2_2') # 3_4_6







    # ==== for mm
    parser.add_argument('--mm_imgfe', type=str, default='resnet18')  # resnet18 convnext_tiny
    parser.add_argument('--mm_imgfe_layers', type=str, default='2_2_2')
    parser.add_argument('--mm_imgfe_planes', type=str, default='64_128_256')
    parser.add_argument('--mm_imgfe_dim', type=int, default=256)
    parser.add_argument('--mm_voxfe_layers', type=str, default='1_1_1')
    parser.add_argument('--mm_voxfe_planes', type=str, default='64_128_256')  
    parser.add_argument('--mm_voxfe_ntd', type=int, default=0)
    parser.add_argument('--mm_voxfe_dim', type=int, default=256)
    parser.add_argument('--mm_bevfe', type=str, default='resnet18')  # resnet18 convnext_tiny
    parser.add_argument('--mm_bevfe_layers', type=str, default='3_3_3')
    parser.add_argument('--mm_bevfe_planes', type=str, default='64_128_256')
    parser.add_argument('--mm_bevfe_dim', type=int, default=256)
    parser.add_argument('--mm_stg2fuse_dim', type=int, default=256) 
    parser.add_argument('--output_type', type=str, default='image_vox_shallow') # image_bev_shallow
    parser.add_argument('--output_l2', type=str, default=True)
    parser.add_argument('--final_type', type=str, default='imageorg_voxorg_shalloworg_stg2image_stg2vox') 
    parser.add_argument('--final_fusetype', type=str, default='add') # add  cat  catadd
    parser.add_argument('--final_l2', type=str, default=False)
    parser.add_argument('--image_embed', type=str, default='stg2image') # imageorg  stg2image
    parser.add_argument('--cloud_embed', type=str, default='stg2vox') # voxorg  stg2vox
    parser.add_argument('--image_weight', type=float, default=1)
    parser.add_argument('--image_learnweight', type=str, default=False)
    parser.add_argument('--bev_weight', type=float, default=1)
    parser.add_argument('--bev_learnweight', type=str, default=False)
    parser.add_argument('--vox_weight', type=float, default=1)
    parser.add_argument('--vox_learnweight', type=str, default=False)
    parser.add_argument('--shallow_weight', type=float, default=1)
    parser.add_argument('--shallow_learnweight', type=str, default=False)
    # 
    parser.add_argument('--diff_type', type=str, default='fcode@relu') # fcode@relu or fcode@sigmoid
    parser.add_argument('--diff_direction', type=str, default='backward') # forward  backward  
    parser.add_argument('--odeint_method', type=str, default='euler')
    parser.add_argument('--odeint_size', type=float, default=0.1)
    parser.add_argument('--sdeint_method', type=str, default='euler') # ito:euler milstein srk
    parser.add_argument('--sdeint_size', type=float, default=0.1)
    parser.add_argument('--cdeint_method', type=str, default='euler') # euler dopri5
    parser.add_argument('--cdeint_size', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--imagevoxorg_weight',  type=float, default=0) 
    parser.add_argument('--imagevoxorg_learnweight', type=str, default=False)
    parser.add_argument('--shalloworg_weight',   type=float, default=1.0) 
    parser.add_argument('--shalloworg_learnweight', type=str, default=False)
    parser.add_argument('--stg2imagevox_weight', type=float, default=0.1) 
    parser.add_argument('--stg2imagevox_learnweight', type=str, default=False)
    parser.add_argument('--stg2fuse_weight',     type=float, default=0)
    parser.add_argument('--stg2fuse_learnweight', type=str, default=False)
    
    parser.add_argument('--stg2gnn', type=str, default='qkv')
    parser.add_argument('--beltrami_k', type=int, default=16)
    parser.add_argument('--stg2nlayers', type=int, default=1)
    
    parser.add_argument('--stg2fuse_type', type=str, default='basic') # for stage 2

    parser.add_argument('--stg2_type', type=str, default='full') # for stage 2
    parser.add_argument('--stg2_useproj', type=str, default=True) # for stage 2
    parser.add_argument('--mm_lossweight', type=str, default='1_0_0') # final cloud image

    parser.add_argument('--otherloss_type', type=str, default='bce') 
    parser.add_argument('--otherloss_weight', type=float, default=0.01)
    parser.add_argument('--tripletloss_weight', type=float, default=1)
    parser.add_argument('--infonceloss_weight', type=float, default=0)






    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit"])
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="gem", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool"])
    # partial  partial_sep
    parser.add_argument("--mining", type=str, default="partial_sep", choices=["partial", "full", "random", "msls_weighted"])

    # Paths parameters
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")

    # Training parameters
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    
    parser.add_argument("--lr_crn_layer", type=float, default=5e-3, help="Learning rate for the CRN layer")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    # Model parameters
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str,help="Path to load checkpoint from, for resuming training or testing.",
                        default=None,
                        # default='logs/ep50_1000_8000__image_bev__satellite_roadmap__4__0.9__outl2False__finall2False__dim256__db256_256__bevrot180__bevjit0.5==__vanilla2d/last_model.pth'
                        )



    # Other parameters
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01,
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true')
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    

    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=0)
    parser.add_argument("--contrast", type=float, default=0)
    parser.add_argument("--saturation", type=float, default=0)
    parser.add_argument("--hue", type=float, default=0)
    parser.add_argument("--rand_perspective", type=float, default=0)
    parser.add_argument("--horizontal_flip", action='store_true')
    parser.add_argument("--random_resized_crop", type=float, default=0)
    parser.add_argument("--random_rotation", type=float, default=0)

    parser.add_argument('--exp_name', type=str, default='none')
















    ##################################### parser
    args = parser.parse_args()
    opt_dict = vars(args)
    # print(args_dict)
    for k, v in opt_dict.items():
        if v in ['False','false']:
            opt_dict[k] = False
        elif v in ['True','true']:
            opt_dict[k] = True
        elif v in ['None','none']:
            opt_dict[k] = None
    args = argparse.Namespace(**opt_dict)

    if args.machine == '4090':
        if args.dataset == 'kitti360': 
            args.dataroot = '/data/sijie/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/data/sijie/radar/nuscenes'
        args.num_workers = 8
    elif args.machine == '4500':
        if args.dataset == 'kitti360': 
            args.dataroot = '/data/sijie/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/data/sijie/radar/nuscenes'
        args.num_workers = 8
    elif args.machine == "sijiewan":
        if args.dataset == 'kitti360': 
            args.dataroot = '/scratch/users/ntu/sijiewan/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/scratch/users/ntu/sijiewan/radar/nuscenes'
        args.num_workers = 16
    elif args.machine == "disheng1":
        if args.dataset == 'kitti360': 
            args.dataroot = '/scratch/users/ntu/disheng0/cmvpr/kitti360/KITTI-360'
        elif args.dataset == 'nuscenes': 
            args.dataroot = '/scratch/users/ntu/disheng0/radar/nuscenes'
        args.num_workers = 16





    #####################################
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


    args.exp_name = ''
    args.exp_name += f'{args.seed}_'
    args.exp_name += f'ep{args.epochs_num}'
    args.exp_name += f'_{args.dataset}'
    args.exp_name += f'_{args.camnames}'
    args.exp_name += f'_{args.cache_refresh_rate}'
    args.exp_name += f'_{args.queries_per_epoch}'
    args.exp_name += f'_{args.maptype}'
    args.exp_name += f'_trbs{args.train_batch_size}'
    args.exp_name += f'_{args.infer_batch_size}'
    args.exp_name += f'_{args.traindownsample}'
    args.exp_name += f'_{args.train_ratio}'
    args.exp_name += f'_sph{args.sph_size}'
    args.exp_name += f'_pc{args.read_pc}'




    #####################################
    args.output_type = args.output_type.split('_')
    args.final_type = args.final_type.split('_')




    #####################################


    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.aggregation == "crn" and args.resume is None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim is not None and args.pca_dataset_folder is None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    if args.backbone == "vit":
        if args.resize != [224, 224] and args.resize != [384, 384]:
            raise ValueError(f'Image size for ViT must be either 224 or 384 {args.resize}')
    if args.backbone == "cct384":
        if args.resize != [384, 384]:
            raise ValueError(f'Image size for CCT384 must be 384, but it is {args.resize}')
    
    if args.backbone in ["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                         "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5"]:
        if args.aggregation in ["cls", "seqpool"]:
            raise ValueError(f"CNNs like {args.backbone} can't work with aggregation {args.aggregation}")
    if args.backbone in ["cct384"]:
        if args.aggregation in ["spoc", "mac", "rmac", "crn", "rrm"]:
            raise ValueError(f"CCT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls, seqpool]")
    if args.backbone == "vit":
        if args.aggregation not in ["cls", "gem", "netvlad"]:
            raise ValueError(f"ViT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls]")



    return args





def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def logging_info(message):
    opt = parse_arguments()
    with open(f'results/{opt.exp_name}.txt', 'a') as f:
        f.write(message + '\n')
        # print(message)
    with open('results.txt', 'a') as f:
        f.write(message + '\n')



def logging_init():
    opt = parse_arguments()
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{opt.exp_name}.txt', 'w') as f:
        f.write(get_datetime())
        f.write('\n')
        f.write(f'{opt.exp_name}\n')
    with open('results.txt', 'w') as f:
        f.write(get_datetime())
        f.write('\n')
        f.write(f'{opt.exp_name}\n')




def logging_end():
    opt = parse_arguments()
    with open(f'results/{opt.exp_name}.txt', 'a') as f:
        f.write('\n')
        f.write(get_datetime())
    with open('results.txt', 'a') as f:
        f.write('\n')
        f.write(get_datetime())



