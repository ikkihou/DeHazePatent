import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger

# from tensorboardX import SummaryWriter
import os
import numpy as np

# from brisque import BRISQUE
import cv2
import random


import time


## plot code by paul
# import seaborn as sns
# import matplotlib.pyplot as plt
import json
from pathlib import Path


# def plot_metric_trend(psnr_metric_dict, ssim_metric_dict, fig_path) -> None:
#     """plot the dehaze metric"""

#     sns.set_theme(style="whitegrid")  # Set Seaborn style

#     ## plot psnr comparision
#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     fig.subplots_adjust(right=0.8)

#     psnr_palette = sns.color_palette("husl", len(psnr_metric_dict))
#     for i, (k, color) in enumerate(zip(psnr_metric_dict.keys(), psnr_palette)):
#         ax1.plot(
#             psnr_metric_dict[k],
#             label=k,
#             marker="o",
#             markersize=6,
#             linestyle="-",
#             color=color,
#         )

#     ax1.set_xlabel("SeqNo")
#     ax1.set_ylabel("psnr")
#     ax1.tick_params(axis="y")

#     ## TODO: complete plotting code
#     plt.savefig(f"{fig_path}/psnr.png")

#     ## plot ssim comparision
#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     fig.subplots_adjust(right=0.8)

#     ssim_palette = sns.color_palette("husl", len(ssim_metric_dict))
#     for i, (k, color) in enumerate(zip(ssim_metric_dict.keys(), ssim_palette)):
#         ax1.plot(
#             ssim_metric_dict[k],
#             label=k,
#             marker="s",
#             markersize=6,
#             linestyle="--",
#             color=color,
#         )

#     ax1.set_xlabel("SeqNo")
#     ax1.set_ylabel("ssim")
#     ax1.tick_params(axis="y")

#     ## TODO: complete plotting code
#     plt.savefig(f"{fig_path}/ssim.png")


seed = 6666
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def calc_mean_rgb(img):
    H, W, C = np.shape(img)
    img = np.reshape(img, (H * W, C))
    return np.mean(img, axis=0)


def fix_img(img, img_ref):
    sr_R, sr_G, sr_B = calc_mean_rgb(img)
    hr_R, hr_G, hr_B = calc_mean_rgb(img_ref)

    R, G, B = sr_R - hr_R, sr_G - hr_G, sr_B - hr_B
    R = np.array(img[:, :, 0]) - R
    G = np.array(img[:, :, 1]) - G
    B = np.array(img[:, :, 2]) - B

    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)

    return np.array(np.concatenate((R, G, B), axis=-1), dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--image_size",
        type=int,
        default=512,
        help="inference image resolution",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/framework_da.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["val"],
        help="val(generation)",
        default="val",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_infer", action="store_true")
    parser.add_argument("-color_fix", default=False)

    # parse configs
    args = parser.parse_args()
    print(args)
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(
        None, opt["path"]["log"], "train", level=logging.INFO, screen=True
    )
    Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))
    # tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt["enable_wandb"]:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "val":
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase
            )  ## raw input dataloader
    logger.info("Initial Dataset Finished")

    # model
    diffusion = Model.create_model(opt)
    logger.info("Initial Model Finished")

    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["val"], schedule_phase="val"
    )

    logger.info("Begin Model Inference.")
    current_step = 0
    current_epoch = 0
    idx = 0
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_psnr2 = 0.0
    avg_ssim2 = 0.0
    avg_infer_time = 0.0

    psnr_metric_dict = {
        "psnr-Dehaze-gt": [],
        "psnr-haze-gt": [],
    }

    ssim_metric_dict = {
        "ssim-Dehaze-gt": [],
        "ssim-haze-gt": [],
    }

    result_path = "{}".format(opt["path"]["results"])
    fig_path = "{}".format(opt["path"]["metric"])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):

        idx += 1

        t1 = time.time()
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        t2 = time.time()
        visuals = diffusion.get_current_visuals(need_LR=False)

        delta_t = t2 - t1

        visuals["SR"] = torch.cat([visuals["SR"], visuals["HR"]], dim=0)

        ## the following two images are 3D(HWC)
        hr_img = Metrics.tensor2img(visuals["HR"])  # uint8
        fake_img = Metrics.tensor2img(visuals["INF"])  # uint8

        sr_img_mode = "grid"
        if sr_img_mode == "single":
            # single img series
            sr_img = visuals["SR"]  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]),
                    "{}/{}_{}_sr_{}.png".format(result_path, current_step, idx, iter),
                )
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals["SR"])  # uint8

            h, w, c = np.shape(hr_img)

            # try:
            #     sr_img[-h-2:-2, -w-2:-2, :] = hr_img
            # except:
            #     pass

            Metrics.save_img(
                sr_img, "{}/{}_{}_sr_process.png".format(result_path, current_step, idx)
            )
            Metrics.save_img(
                Metrics.tensor2img(visuals["SR"][-2]),
                "{}/{}_{}_sr.png".format(result_path, current_step, idx),
            )

        Metrics.save_img(
            hr_img, "{}/{}_{}_hr.png".format(result_path, current_step, idx)
        )
        Metrics.save_img(
            fake_img, "{}/{}_{}_inf.png".format(result_path, current_step, idx)
        )

        sr_img = Metrics.tensor2img(visuals["SR"][-2])
        if args.color_fix:
            # print(sr_img)
            # print(fake_img)
            # print(sr_img.shape)
            # print(fake_img.shape)

            sr_img = fix_img(sr_img, fake_img)
            # cv2.imwrite('{}/{}_{}_sr.png'.format(result_path, current_step, idx), sr_img)

        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        psnr2 = Metrics.calculate_psnr(fake_img, hr_img)
        ssim2 = Metrics.calculate_ssim(fake_img, hr_img)
        psnr_metric_dict["psnr-Dehaze-gt"].append(psnr)
        ssim_metric_dict["ssim-Dehaze-gt"].append(ssim)
        psnr_metric_dict["psnr-haze-gt"].append(psnr2)
        ssim_metric_dict["ssim-haze-gt"].append(ssim2)
        # brisque = BRISQUE('{}/{}_{}_sr.png'.format(result_path, current_step, idx)).score()
        brisque = 0

        avg_psnr += psnr
        avg_ssim += ssim
        avg_psnr2 += psnr2
        avg_ssim2 += ssim2
        avg_infer_time += delta_t

        print(
            f"psnr: {psnr}, ssim:{ssim}, save to {'{}/{}_{}_sr_process.png'.format(result_path, current_step, idx)}"
        )

        if wandb_logger and opt["log_infer"]:
            wandb_logger.log_eval_data(
                fake_img, Metrics.tensor2img(visuals["SR"][-1]), hr_img
            )

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_psnr2 = avg_psnr2 / idx
    avg_ssim2 = avg_ssim2 / idx
    avg_infer_time = avg_infer_time / idx

    ## code by paul
    # plot_metric_trend(psnr_metric_dict, ssim_metric_dict, fig_path)
    with open(str(Path(fig_path, "psnr_metric.json")), "w") as f:
        json.dump(psnr_metric_dict, f)

    with open(str(Path(fig_path, "ssim_metric.json")), "w") as f:
        json.dump(ssim_metric_dict, f)

    print(
        f"avg_psnr: {avg_psnr}, avg_ssim: {avg_ssim}, avg_psnr2: {avg_psnr2}, avg_ssim: {avg_ssim2}, avg_infer_time: {avg_infer_time}"
    )

    if wandb_logger and opt["log_infer"]:
        wandb_logger.log_eval_table(commit=True)
