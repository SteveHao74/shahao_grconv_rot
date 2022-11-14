import datetime
import os
import sys
import argparse
import logging

import cv2

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX
from pathlib import Path

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output

logging.basicConfig(level=logging.INFO)

# cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
GMDATA_PATH = Path("/media/shahao/F07EE98F7EE94F42/win_stevehao/Research/gmdata")
DATASET_PATH= GMDATA_PATH.joinpath('datasets')
INPUT_DATA_PATH = DATASET_PATH.joinpath('train_datasets/gg_data/jaq')#small_data/cor#gmd/noisy_gmd
# DATASET_PATH = GMDATA_PATH.joinpath('datasets/train_datasets')
# INPUT_DATA_PATH = DATASET_PATH.joinpath('gg_data/shahao_data/gmd')#gg_data/shahao_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')
    parser.add_argument('--resume', type=str, default=False,
                        help='to resume the interrupted model training')
    parser.add_argument('--start-epoch', type=int, default=5,
                        help='the next epoch to start')
    parser.add_argument('--resume-path', type=str, default="output/models/220222_2333_no_normal_gauss_jaq/epoch_04_iou_0.89",
                        help='to resume the interrupted model training')    
    # Network
    parser.add_argument('--network', type=str, default='grconvnet3', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default="jacquard",help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default=INPUT_DATA_PATH,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')#0.001#gmd0.0001
    parser.add_argument('--split', type=float, default=0.99, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1416, help='Batches per Epoch')#jaq1416gmd1426
    parser.add_argument('--val-batches', type=int, default=50, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--description', type=str, default='no_normal_gauss_jaq', help='Training description')#gauss_cor
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--logdir', type=str, default='tensorboard/', help='Log directory')
    parser.add_argument('--vis', action='store_true', default=False,help='Visualise the training process')

    args = parser.parse_args()
    return args

# Anti-Clockwise rotate
def rotate_state(state,a,device):
    angle = a 
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float,device=device)
    grid = F.affine_grid(theta.unsqueeze(0), state.size())
    output = F.grid_sample(state, grid)
    return output


def validate(policy_net, device, val_data, batches_per_epoch,epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    policy_net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break
                # import pdb; pdb.set_trace()
                xc = x.transpose(1, 0).to(device)
                # for i in range(9):
                #     plt.imshow(xc[i].cpu().squeeze(0).squeeze(0).numpy())
                #     plt.show()
                yc = []
                for i in range(len(y)):
                    y[i] = y[i].transpose(1, 0).to(device)
                    yc.append(y[i])

                # xc = []
                # for i in range(9):
                #     xc.append(rotate_state(x.to(device),math.pi/2-i*math.pi/9,device))
                #     # plt.imshow(xc[i].cpu().numpy()[0][0])
                #     # plt.show()
                # xc = torch.cat(xc)
                #
                # # ############################
                # # if batch_idx == 200:
                # #     plt.imshow(x.numpy()[0])
                # #     plt.show()
                #
                # yc = []
                # for j in range(len(y)):
                #     yjc=[]
                #     for i in range(9):
                #         yjc.append(rotate_state(y[j].to(device),math.pi/2-i*math.pi/9,device))
                #     yc.append(torch.cat(yjc))

                lossd = policy_net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item()/ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item()/ld

                q_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['width'])

                # ############################
                # if batch_idx ==200:
                #     for i in range(9):
                #         plt.imshow(q_out[i])
                #         plt.show()
                #         plt.imshow(w_out[i])
                #         plt.show()
                ############################
                # if batch_idx % 20 == 0:
                #     np.savez('/home/abb/Pictures/Dataset_SIH/GGCNN_rot/' + str(epoch)+ '_' + str(batch_idx) + '.npz', \
                #              d_img=xc.cpu().squeeze(1).numpy(), \
                #              q_img=q_out)

                s,gs = evaluation.calculate_iou_match_rot(q_out,
                                                   val_data.dataset.get_gtbb_val(didx, rot, zoom_factor),
                                                   no_grasps=1,
                                                   grasp_width=w_out,
                                                   zoom_factor=zoom_factor
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
                    # q_gt,_,_=val_data.dataset.get_gtbb_val(didx, rot, zoom_factor).draw((300, 300))
                    # np.savez(
                    #     '/home/abb/Pictures/Dataset_SIH/GGCNN_rot/' + str(epoch) + '_' + str(batch_idx) + '.npz', \
                    #     d_img=xc.cpu().squeeze(1).numpy(), \
                    #     q_img=q_out,
                    #     q_gt= q_gt
                    # )

    return results


def train(epoch, policy_net,device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    policy_net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx < batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            # pos_output, cos_output, sin_output, width_output\
            #     = target_net(xc)
            yc = []
            for i in range(len(y)):
                y[i]=y[i].to(device)
                yc.append(y[i])

            # pos_output[torch.where(y[0] != 0)] = y[0][torch.where(y[0] != 0)]
            # yc.append(pos_output.to(device))
            # cos_output[torch.where(y[0] != 0)] = y[1][torch.where(y[0] != 0)]
            # yc.append(cos_output.to(device))
            # sin_output[torch.where(y[0] != 0)] = y[2][torch.where(y[0] != 0)]
            # yc.append(sin_output.to(device))
            # width_output[torch.where(y[0] != 0)] = y[3][torch.where(y[0] != 0)]
            # yc.append(width_output.to(device))

            lossd = policy_net.compute_loss(xc, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            # if vis:
            #     imgs = []
            #     n_img = min(4, x.shape[0])
            #     for idx in range(n_img):
            #         imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
            #             x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in lossd['pred'].values()])
            #     gridshow('Display', imgs,
            #              [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
            #              [cv2.COLORMAP_BONE] * 10 * n_img, 10)
            #     cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()
    # Set-up output directories
    if args.resume : 
        net_desc= args.resume_path.split("/")[-2]
    else:
        dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
        net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))


    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    train_dataset = Dataset(args.dataset_path, start=0, end=args.split, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    
    print("沙昊1",len(train_dataset.grasp_files))
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=True,
                          include_depth=args.use_depth, include_rgb=args.use_rgb,is_training=False)


    print("沙昊",len(val_dataset.grasp_files))   
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=args.num_workers
        num_workers = args.num_workers
    )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1*args.use_depth + 3*args.use_rgb
    target_net = None
    if args.resume :
        policy_net = torch.load(args.resume_path)
        start_epoch = args.start_epoch
        
    else :
        ggcnn = get_network(args.network)
        policy_net = ggcnn(input_channels=input_channels,dropout=args.use_dropout,
            prob=args.dropout_prob,
            channel_size=args.channel_size)
        # target_net = ggcnn(input_channels=input_channels)
        start_epoch = 0
    
    device = torch.device("cuda:0")
    policy_net = policy_net.to(device)
    # target_net = target_net.to(device)
    # target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(),lr=args.lr)#lr=0.0001
    # optimizer = optim.SGD(policy_net.parameters(), lr=0.0001, momentum=0.09)
    logging.info('Done')

    # Print model architecture.
    summary(policy_net, (input_channels, 300, 300))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(policy_net, (input_channels, 300, 300))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(start_epoch,args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, policy_net,  device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        ##########################################################
        # if epoch%3==0:
        #     target_net.load_state_dict(policy_net.state_dict())
        # target_net.load_state_dict(policy_net.state_dict())

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # Run Validation
        logging.info('Validating...')
        test_results = validate(policy_net, device, val_data, args.val_batches,epoch)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct']/(test_results['correct']+test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 3) == 0:
            torch.save(policy_net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
