# the script plots the training loss and accuracy change over the training process
import os
import torch
import matplotlib.pyplot as plt
from matplotlib import style


def save_training_curves(mode, all_checkpoint_paths, output_dir):
    # initialize
    all_train_losses = {}
    all_val_losses = {}
    all_train_accs = {}
    all_val_accs = {}

    for cur_model_type, cur_checkpoint_path in all_checkpoint_paths.items():
        cur_checkpoint = torch.load(cur_checkpoint_path)
        history = cur_checkpoint["history"]
        all_train_losses[cur_model_type] = history[f"train_{mode}_loss"]
        all_train_accs[cur_model_type] = history[f"train_{mode}_accuracy"]
        all_val_accs[cur_model_type] = history[f"val_{mode}_accuracy"]
        all_val_losses[cur_model_type] = history[f"val_{mode}_loss"]

    # plot the train and val losses
    # colors indicate train or val, line styles indicate different models
    line_styles = ["solid", "dotted", "dashed", "dashdot"]
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    for i, cur_name in enumerate(all_checkpoint_paths.keys()):
        cur_train_loss = all_train_losses[cur_name]
        cur_val_loss = all_val_losses[cur_name]
        ax.plot(
            cur_train_loss,
            color="orange",
            linestyle=line_styles[i],
            label=f"{cur_name}_train",
        )
        ax.plot(
            cur_val_loss,
            color="green",
            linestyle=line_styles[i],
            label=f"{cur_name}_val",
        )
    plt.legend()
    loss_curve_path = os.path.join(output_dir, f"{mode}_loss_curve.png")
    plt.savefig(loss_curve_path)
    print(f"{mode} loss curves have been saved to {loss_curve_path}")

    # plot the train and val accuracies
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    for i, cur_name in enumerate(all_checkpoint_paths.keys()):
        cur_train_accs = all_train_accs[cur_name]
        cur_val_accs = all_val_accs[cur_name]
        ax.plot(
            cur_train_accs,
            color="orange",
            linestyle=line_styles[i],
            label=f"{cur_name}_train",
        )
        ax.plot(
            cur_val_accs,
            color="green",
            linestyle=line_styles[i],
            label=f"{cur_name}_val",
        )
    plt.legend()
    acc_curve_path = os.path.join(output_dir, f"{mode}_acc_curve.png")
    plt.savefig(acc_curve_path)
    print(f"{mode} accuracy curves have been saved to {acc_curve_path}")


output_dir = "./vl_model/plots/"
# each checkpoint path corresponds to a model
all_checkpoint_paths = {
    # "Baseline-Finetuning": "./MobileOne/finetuned_models_data_v2/mobileone_s4_finetune_adamW_epoch_50.pt",
    "Frozen_Text_Encoder": "/home/ec2-user/SageMaker/Violence-Detector/vl_model/models/frozen_text_encoder/frozen_vlsm_mobileone-s4_distilbert_adamw_batch_size_128_epoch_50.pt",
    "Unfrozen_Text_Encoder": "/home/ec2-user/SageMaker/Violence-Detector/vl_model/models/unfrozen_text_encoder/unfrozen_vlsm_mobileone-s4_distilbert_adamw_batch_size_128_epoch_50.pt",
    # "inference_option_1": "./vl_model/models/frozen_text_encoder/vlsm_mobileone-s4_distilbert_adamW_epoch_50.pt",
}

# plot training curves based on epochs
save_training_curves("epoch", all_checkpoint_paths, output_dir)

# plot training curves based on batches/steps
save_training_curves("batch", all_checkpoint_paths, output_dir)
