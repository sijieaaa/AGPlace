



import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')


from tools.options import parse_arguments
opt = parse_arguments()








def compute_bcemat(eastnorthdist_mat, positive_thd=10, negative_thd=25):
    bcemat = torch.zeros_like(eastnorthdist_mat, dtype=torch.float32)-1 # default -1
    bcemat[eastnorthdist_mat < positive_thd] = 0 # 0 for closer, 1 for farther
    bcemat[eastnorthdist_mat > negative_thd] = 1 # 0 for closer, 1 for farther

    return bcemat




def compute_loss(featsdist_mat, bcemat, otherloss_type):
    mask = (bcemat != -1)
    sum_mask = mask.sum()
    if otherloss_type == 'bce':
        featsdist_mat = featsdist_mat[mask]
        bcemat = bcemat[mask]
        loss = nn.BCEWithLogitsLoss()(input=featsdist_mat, target=bcemat)
    elif otherloss_type == 'mse':
        featsdist_mat = featsdist_mat[mask]
        featsdist_mat = torch.sigmoid(featsdist_mat)
        bcemat = bcemat[mask]
        loss = nn.MSELoss()(input=featsdist_mat, target=bcemat)
    elif otherloss_type == 'l1':
        featsdist_mat = featsdist_mat[mask]
        featsdist_mat = torch.sigmoid(featsdist_mat)
        bcemat = bcemat[mask]
        loss = nn.L1Loss()(input=featsdist_mat, target=bcemat)
    else:
        raise NotImplementedError

    return loss


def compute_other_loss(feats_ground, feats_aerial, data_dict, positive_thd=10, negative_thd=25):

    feats_Gembed = feats_ground['embedding'] # [b,c]
    feats_Gimageorg = feats_ground['imagevec_org']
    feats_Gvoxorg = feats_ground['voxvec_org']

    feats_Aembed = feats_aerial['embedding']
    b,ndb,c = feats_Aembed.shape
    feats_Aembed = feats_Aembed.view(-1, c) # [b*ndb,c]

    eastnorth_G = data_dict['query_eastnorth'] # [b,2]
    eastnorth_A = data_dict['db_eastnorth'] # [b,ndb,2]
    eastnorth_A = eastnorth_A.view(-1, 2) # [b*ndb,2]
    eastnorth_AG = torch.cat([eastnorth_A, eastnorth_G], dim=0) # [b+b*ndb,2]

    # Loss between AG-AG
    feats_Aembed 
    feats_AembedGembed = torch.cat([feats_Aembed, feats_Gembed], dim=0) # [b+b*ndb,c]
    feats_AembedGimageorg = torch.cat([feats_Aembed, feats_Gimageorg], dim=0) # [b+b*ndb,c]
    feats_AembedGvoxorg = torch.cat([feats_Aembed, feats_Gvoxorg], dim=0) # [b+b*ndb,c]
    feats_Gembed
    feats_Gimageorg
    feats_Gvoxorg

    featsdist_Aembed_Aembed = torch.cdist(feats_Aembed, feats_Aembed) 
    featsdist_Gembed_AembedGembed = torch.cdist(feats_Gembed, feats_AembedGembed)
    featsdist_Gimageorg_AembedGimageorg = torch.cdist(feats_Gimageorg, feats_AembedGimageorg)
    featsdist_Gvoxorg_AembedGvoxorg = torch.cdist(feats_Gvoxorg, feats_AembedGvoxorg)
    
    eastnorthdist_Aembed_Aembed = torch.cdist(eastnorth_A, eastnorth_A)
    eastnorthdist_Gembed_AembedGembed = torch.cdist(eastnorth_G, eastnorth_AG)
    eastnorthdist_Gimageorg_AembedGimageorg = torch.cdist(eastnorth_G, eastnorth_AG)
    eastnorthdist_Gvoxorg_AembedGvoxorg = torch.cdist(eastnorth_G, eastnorth_AG)

    assert featsdist_Aembed_Aembed.shape == eastnorthdist_Aembed_Aembed.shape
    assert featsdist_Gembed_AembedGembed.shape == eastnorthdist_Gembed_AembedGembed.shape
    assert featsdist_Gimageorg_AembedGimageorg.shape == eastnorthdist_Gimageorg_AembedGimageorg.shape
    assert featsdist_Gvoxorg_AembedGvoxorg.shape == eastnorthdist_Gvoxorg_AembedGvoxorg.shape

    # Compute BCE matrix
    bcemat_Aembed_Aembed = compute_bcemat(eastnorthdist_Aembed_Aembed, positive_thd, negative_thd)
    bcemat_Gembed_AembedGembed = compute_bcemat(eastnorthdist_Gembed_AembedGembed, positive_thd, negative_thd)
    bcemat_Gimageorg_AembedGimageorg = compute_bcemat(eastnorthdist_Gimageorg_AembedGimageorg, positive_thd, negative_thd)
    bcemat_Gvoxorg_AembedGvoxorg = compute_bcemat(eastnorthdist_Gvoxorg_AembedGvoxorg, positive_thd, negative_thd)



    loss_Aembed_Aembed = compute_loss(featsdist_Aembed_Aembed, bcemat_Aembed_Aembed, opt.otherloss_type)
    loss_Gembed_AembedGembed = compute_loss(featsdist_Gembed_AembedGembed, bcemat_Gembed_AembedGembed, opt.otherloss_type)
    loss_Gimageorg_AembedGimageorg = compute_loss(featsdist_Gimageorg_AembedGimageorg, bcemat_Gimageorg_AembedGimageorg, opt.otherloss_type)
    loss_Gvoxorg_AembedGvoxorg = compute_loss(featsdist_Gvoxorg_AembedGvoxorg, bcemat_Gvoxorg_AembedGvoxorg, opt.otherloss_type)

    loss = (
        + loss_Aembed_Aembed * opt.otherloss_weight  
        + loss_Gembed_AembedGembed * opt.otherloss_weight 
        + loss_Gimageorg_AembedGimageorg * opt.otherloss_weight 
        + loss_Gvoxorg_AembedGvoxorg * opt.otherloss_weight
    )

    return loss




if __name__ == '__main__':

    feats_ground = torch.load('feats_ground.pth')
    feats_aerial = torch.load('feats_aerial.pth')
    data_dict = torch.load('data_dict.pth')

    loss = compute_other_loss(feats_ground, feats_aerial, data_dict)