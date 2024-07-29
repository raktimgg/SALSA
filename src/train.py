import gc
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data.sejong_southbay import SejongSouthbayTupleLoader

from models.salsa import SALSA

from loss.loss import find_loss
from utils.misc_utils import tuple_collate_fn, read_yaml_config

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')
    del model_parameters, params

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def main():
    config = read_yaml_config(os.path.join(os.path.dirname(__file__),'config/train.yaml'))
    writer = SummaryWriter(config['writer_loc'])
    # Get data loader
    batch_size = config['batch_size']
    train_transform = None
    dataset = SejongSouthbayTupleLoader(cached_queries=config['cached_queries'], pcl_transform=train_transform)
    device = config['device']

    model = SALSA(voxel_sz=0.5).to(device)

    model.train()
    print_nb_params(model)
    print_model_size(model)
    MAX_EPOCH = config['max_epoch']

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])


    kk_batch = 0
    kk_subcache = 0
    for e in range(MAX_EPOCH):
        EPOCH_LOSS = []
        time1 = time.time()
        dataset.new_epoch()
        steps_per_epoch = int(np.ceil(1000/batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['max_lr'],epochs=dataset.nCacheSubset, steps_per_epoch=steps_per_epoch,anneal_strategy='cos', cycle_momentum=False)
        lr_list = [scheduler.get_last_lr()]
        for ii in range(dataset.nCacheSubset):
            scheduler.step((ii+1)*steps_per_epoch)
            lr_list.append(scheduler.get_last_lr())

        for current_subset in range(0,dataset.nCacheSubset):
            CACHE_LOSS = []

            dataset.current_subset=current_subset
            dataset.update_subcache(model,outputdim=config['outdim'])
            if len(dataset.triplets)==0:
                continue
            model.train()
            data_loader = torch.utils.data.DataLoader(dataset=dataset,shuffle=True, batch_size=batch_size,collate_fn=tuple_collate_fn, num_workers=16)
            scheduler_lr = np.linspace(lr_list[current_subset],lr_list[current_subset+1],len(data_loader))
            
            for i, batch_data in enumerate(data_loader):
                model.zero_grad()
                optimizer.zero_grad()
                coord, xyz, feat, batch_number, labels, point_pos_pairs = batch_data
                coord, xyz, feat, batch_number, labels = coord.to(device), xyz.to(device), feat.to(device), batch_number.to(device),labels.to(device)
                local_features, global_descriptor = model(coord, xyz, feat, batch_number)

                loss = find_loss(local_features, global_descriptor, point_pos_pairs)
                loss.backward()
                optimizer.step()
                for param_group in optimizer.param_groups:
                    last_lr = param_group['lr']
                    param_group['lr'] = scheduler_lr[i][0]
                writer.add_scalar("Batch Loss", loss.item(), kk_batch)
                writer.add_scalar("Batch LR", last_lr, kk_batch)
                kk_batch += 1
                CACHE_LOSS.append(loss.item())
                sys.stdout.write('\r' + 'Epoch ' + str(e + 1) + ' / ' + str(MAX_EPOCH) + ' Subset ' + str(current_subset + 1) + ' / ' + str(dataset.nCacheSubset) + ' Progress ' + str(i+1) + ' / ' + str(len(data_loader))+ ' Loss ' + str(format(loss.item(),'.2f')) + ' time '+ str(format(time.time()-time1,'.2f'))+' seconds.')

            torch.save(model.state_dict(),os.path.join(os.path.dirname(__file__),'checkpoints/SALSA/Model/model_'+str(e)+'.pth'))
            del coord, xyz, feat, batch_number, labels,  local_features, global_descriptor, point_pos_pairs
            gc.collect()
            torch.cuda.empty_cache()
            cache_loss_avg = sum(CACHE_LOSS)/len(CACHE_LOSS)*steps_per_epoch

            writer.add_scalar("Subcache Loss", cache_loss_avg, kk_subcache)
            writer.add_scalar("Subcache LR", last_lr, kk_subcache)
            kk_subcache += 1

            EPOCH_LOSS.append(cache_loss_avg)
            print(' ')
            print('Avg. Subcache Loss', cache_loss_avg)
        torch.save(model.state_dict(),os.path.join(os.path.dirname(__file__),'checkpoints/SALSA/Model/model_'+str(e)+'.pth'))
        epoch_loss_avg = sum(EPOCH_LOSS)/len(EPOCH_LOSS)
        print(' ')
        print('Avg. EPOCH Loss', epoch_loss_avg)
        writer.add_scalar("Epoch Loss", epoch_loss_avg, e)


if __name__ == "__main__":
    seed = 1100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    gc.collect()
    main()


