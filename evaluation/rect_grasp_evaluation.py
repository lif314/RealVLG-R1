import torch
import time
import argparse
import logging
from evaluation.grasp import calculate_iou_match

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--model_path', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Path to saved networks to evaluate')
    
    # Evaluation
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    args = parser.parse_args()
    return args

def evaluate_rect_grasp():

    args = parse_args()

    results = {'correct': 0, 'failed': 0}


    start_time = time.time()

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom, prompt, query) in enumerate(test_data):
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc, prompt, query)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                    no_grasps=args.n_grasps,
                                                    grasp_width=width_img,
                                                    threshold=args.iou_threshold
                                                    )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            if args.vis:
                save_results(
                    rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                    depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=args.n_grasps,
                    grasp_width_img=width_img
                )

    avg_time = (time.time() - start_time) / len(test_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                    results['correct'] + results['failed'],
                                                    results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))

    torch.cuda.empty_cache()