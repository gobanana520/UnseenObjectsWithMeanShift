#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""
import os
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import argparse
import torch

import _init_paths
from fcn.config import cfg, cfg_from_file
from fcn.test_utils import test_sample_hand_object_data
from fcn.test_demo import get_predictor, get_predictor_crop

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.abspath(os.path.join(CURR_DIR, "../../../data/calibration"))

RS_CAMERAS = [
    "105322251564",
    "043422252387",
    # "037522251142",
    "105322251225",
    "108222250342",
    "117222250549",
    "046122250168",
    # "115422250549",
]


# def compute_xyz(depth_img, fx, fy, px, py, height, width):
#     indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
#     z_e = depth_img
#     x_e = (indices[..., 1] - px) * z_e / fx
#     y_e = (indices[..., 0] - py) * z_e / fy
#     xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
#     return xyz_img


def compute_xyz(depth_img, K_matrix):
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def read_sample(color_file, depth_file, intrinsic_file):
    # bgr image
    color_img = cv2.imread(color_file).astype(np.float32) / 255.0
    # depth image
    depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    # intrinsic matrix
    K = load_intrinsic_matrix_from_yaml(intrinsic_file)

    xyz_img = compute_xyz(depth_img, K)

    color_tensor = torch.from_numpy(color_img)
    pixel_mean = torch.tensor(
        np.array([[[102.9801, 115.9465, 122.7717]]]) / 255.0
    ).float()
    color_tensor -= pixel_mean
    image_blob = color_tensor.permute(2, 0, 1)
    depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
    sample = {
        "image_color": image_blob,
        "depth": depth_blob,
        "file_name": color_file,
    }
    return sample


def load_data_from_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def load_meta_data(file_path):
    data = load_data_from_yaml(file_path)
    rs_serials = data["realsense"]["rs_serials"]
    rs_width = data["realsense"]["rs_width"]
    rs_height = data["realsense"]["rs_height"]
    rs_count = data["realsense"]["rs_count"]
    return rs_serials, rs_width, rs_height, rs_count


def load_intrinsic_matrix_from_yaml(file_path, key="color"):
    data = load_data_from_yaml(file_path)
    K = np.array(
        [
            [data[key]["fx"], 0, data[key]["ppx"]],
            [0, data[key]["fy"], data[key]["ppy"]],
            [0, 0, 1],
        ]
    )
    return K


def parse_args():
    parser = argparse.ArgumentParser(description="Test a PoseCNN network")

    parser.add_argument("--cfg", dest="cfg_file", help="optional config file", type=str)
    parser.add_argument(
        "--sequence_folder",
        dest="sequence_folder",
        help="path of the rosbag extracted folder",
        type=str,
    )
    parser.add_argument(
        "--save_folder_name",
        dest="save_folder_name",
        help="name of the save folder",
        type=str,
    )
    parser.add_argument(
        "--gpu", dest="gpu_id", help="GPU id to use", default=0, type=int
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="initialize with pretrained checkpoint",
        default=os.path.join(CURR_DIR, "../data/checkpoints/norm_model_0069999.pth"),
        type=str,
    )
    parser.add_argument(
        "--pretrained_crop",
        dest="pretrained_crop",
        help="initialize with pretrained checkpoint for crops",
        default=os.path.join(CURR_DIR, "../data/checkpoints/crop_dec9_model_final.pth"),
        type=str,
    )
    parser.add_argument(
        "--network_cfg",
        dest="network_cfg_file",
        help="config file for first stage network",
        default=os.path.join(CURR_DIR, "../MSMFormer/configs/tabletop_pretrained.yaml"),
        type=str,
    )
    parser.add_argument(
        "--network_crop_cfg",
        dest="network_crop_cfg_file",
        help="config file  for second stage network",
        default=os.path.join(
            CURR_DIR, "../MSMFormer/configs/crop_tabletop_pretrained.yaml"
        ),
        type=str,
    )
    parser.add_argument(
        "--rand",
        dest="randomize",
        help="randomize (do not use a fixed seed)",
        action="store_true",
    )
    parser.add_argument(
        "--input_image",
        dest="input_image",
        help="the type of image",
        default="RGBD_ADD",
        type=str,
    )
    args = parser.parse_args()
    return args


def get_masks(prediction, score):
    masks = np.zeros_like(prediction, dtype=np.uint8)
    best_score = np.max(score)
    masks[score == best_score] = 255
    return masks


def main(cfg):
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device("cuda:{:d}".format(cfg.gpu_id))
    cfg.instance_id = 0
    cfg.MODE = "TEST"
    print("GPU device {:d}".format(args.gpu_id))

    # prepare network
    predictor, cfg = get_predictor(
        cfg_file=args.network_cfg_file,
        weight_path=args.pretrained,
        input_image=args.input_image,
    )
    predictor_crop, cfg_crop = get_predictor_crop(
        cfg_file=args.network_crop_cfg_file,
        weight_path=args.pretrained_crop,
        input_image=args.input_image,
    )

    ###########################################################################
    # Process the sequence
    ###########################################################################
    sequence_folder = args.sequence_folder
    meta_file = os.path.join(sequence_folder, "meta.yaml")
    rs_serials, rs_width, rs_height, rs_count = load_meta_data(meta_file)

    for rs_serial in tqdm(RS_CAMERAS):
        if rs_serial not in rs_serials:
            continue
        tqdm.write(f"Processing {rs_serial}")
        save_folder = os.path.join(
            sequence_folder,
            "./data_processing/unseen",
            args.save_folder_name,
            rs_serial,
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        intrinsic_file = os.path.join(
            CALIB_DIR, "intrinsics", f"rs_{rs_serial}_{rs_width}x{rs_height}.yaml"
        )

        for idx in range(1):
            color_file = os.path.join(
                sequence_folder, rs_serial, f"color_{idx:06d}.jpg"
            )
            depth_file = os.path.join(
                sequence_folder, rs_serial, f"depth_{idx:06d}.png"
            )
            sample = read_sample(color_file, depth_file, intrinsic_file)
            (
                prediction,
                prediction_refined,
                prediction_score,
            ) = test_sample_hand_object_data(
                cfg,
                sample,
                predictor,
                predictor_crop,
                visualization=False,
                topk=False,
                # topk=True,
                confident_score=0.6,
                save_image=False,
                save_folder=save_folder,
            )

            results = {
                "prediction": prediction,
                "prediction_refined": prediction_refined,
                "prediction_score": prediction_score,
            }
            results_name = os.path.join(save_folder, f"results_{idx:06d}.npz")
            np.savez(results_name, **results)

            # segmentation labels
            img_mask = prediction_refined[0]
            filename = os.path.join(save_folder, f"color_{idx:06d}_mask.png")
            cv2.imwrite(filename, img_mask)


if __name__ == "__main__":
    main(cfg)
