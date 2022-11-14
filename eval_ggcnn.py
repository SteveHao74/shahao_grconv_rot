import argparse
import logging

import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import numpy as np
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    # args.network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200514_1906_anglesless10_withoutTnn_rot/epoch_10_iou_0.50'
    # args.network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200517_1501_anglesless10_withoutTnn_rot_12angle_mindistance5_70width/epoch_01_iou_0.38'
    args.network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200518_1650_anglesless10_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_27_iou_0.55'
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb,is_training=False)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            # xc = x.to(device)
            # yc = [yi.to(device) for yi in y]
            xc = x.transpose(1, 0).to(device)
            yc = []
            for i in range(len(y)):
                y[i] = y[i].transpose(1, 0).to(device)
                yc.append(y[i])

            lossd = net.compute_loss(xc, yc)

            q_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['width'])

            if args.iou_eval:
                s,gs = evaluation.calculate_iou_match_rot(q_img, test_data.dataset.get_gtbb_val(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   zoom_factor=zoom
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1
                    print(idx)

                d_img = xc[6].cpu().squeeze(0).squeeze(0).numpy().copy()
                crop_img = d_img.copy()
                # plt.imshow(crop_img)
                # plt.show()
                for gr in gs:
                    # g = gr.as_grasp
                    gr = gr.as_gr
                    # cv2.circle(crop_img, (g.center[1],g.center[0]),2,(0, 0, 255))
                    cv2.line(crop_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
                             (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
                             (255, 0, 0), 2)
                    cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
                             (255, 0, 0), 2)
                    cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
                             (255, 0, 0), 2)
                # plt.imshow(crop_img,alpha=0.8)
                # # plt.imshow(q_img,alpha=0.5)
                # plt.show()
                q_gt, _, _ = test_data.dataset.get_gtbb_val(didx, rot, zoom).draw((300, 300))
                np.savez(
                    '/home/abb/Pictures/Dataset_SIH/GGCNN_rot2/'  + str(idx) + '.npz', \
                    d_img=xc.cpu().squeeze(1).numpy(), \
                    q_img=q_img,
                    gs = crop_img,
                    q_gt= q_gt
                )

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                       test_data.dataset.get_depth(didx, rot, zoom), q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))