import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loss.global_loss import triplet_margin_loss
from loss.local_consistency_loss import point_contrastive_loss

def find_loss(local_features, global_descriptors,point_pos_pairs):
    global_loss = 0
    local_loss = 0
    batch_size = int(global_descriptors.shape[0]/3)
    for i in range(batch_size):

        ## Global descriptor triplet loss ###################################################
        q_gd = global_descriptors[i][None,...]
        p_gd = global_descriptors[(1*batch_size) + i][None,...]
        n_gd = global_descriptors[(2*batch_size) + i][None,...]


        global_loss += triplet_margin_loss(q_gd, p_gd, n_gd, margin=0.1)

        ## Local features loss ###############################################################
        ## Point triplet loss ###################
        if point_pos_pairs[i].shape[0]>0:
            point_loss = point_contrastive_loss(local_features[i], local_features[(1*batch_size) + i], point_pos_pairs[i])
        else:
            point_loss = 0
        local_loss += point_loss
        #########################################
    # loss = global_loss + local_loss
    loss = global_loss + local_loss
    return loss