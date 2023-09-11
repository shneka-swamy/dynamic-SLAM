import argparse
from pathlib import Path

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='FlowFormer')
    # Arguments for the optical flow model
    parser.add_argument('--eval_type', default='tum', help='the dataset used for evaluation is TUM')
    parser.add_argument('--root_dir', default=Path('..'))
    parser.add_argument('--seq_dir', default=Path('Datasets/rgbd_dataset_freiburg3_walking_halfsphere/rgb'))
    parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=1200)    # ending index of the image sequence
    parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.

    # Arguments for the segmentation model
    parser.add_argument('--trained_model',
                        default='yolact/models/yolact_base_54_800000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    

    # For visualisation and evaluation
    parser.add_argument('--only_video', action='store_true', help='only generate video')
    parser.add_argument('--run_full_system', action='store_true', help='run the full system')
    parser.add_argument('--run_flowformer', action='store_true', help='run the flowformer')
    parser.add_argument('--output_dir', default=Path('Output'), help='output directory')
    parser.add_argument('--seg_output_dir', default=Path('Output/Segmentation'), help='segmentation output directory')
    parser.add_argument('--run_yolact', action='store_true', help='run the segmentation model')
    parser.add_argument('--run_blur_detection', action='store_true', help='run the blur detection model')
    parser.add_argument('--run_tracker', action='store_true', help='run the tracker')
    parser.add_argument('--run_contour', action='store_true', help='run the contour detection')
    parser.add_argument('--run_homography', action='store_true', help='run the homography estimation')


    return parser.parse_args()


def main():
    args = argparser()
    return args


if __name__ == '__main__':
    main()