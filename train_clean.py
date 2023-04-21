import os
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, PreActResNet34
from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_step(
    netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, epoch, opt, tf_writer
):
    netC.train()
    netG.train()
    print(" Training:")
    total = 0
    total_clean = 0
    total_correct_clean = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1) in enumerate(train_dl1):
        optimizerC.zero_grad()
        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        bs = inputs1.shape[0]
        total_inputs = inputs1
        total_targets = targets1

        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        total_loss = loss_ce
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

        total += bs
        total_clean += bs

        total_correct_clean += torch.sum(torch.argmax(preds, dim=1) == total_targets)

        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        if not batch_idx % 50:
            print(batch_idx, len(train_dl1), "CE loss: {:.4f} - Clean Accuracy: {:.3f}".format(
                avg_loss, acc_clean))

        # Saving images for debugging

        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            images = netG.denormalize_pattern(inputs1[:25])
            file_name = "{}_{}_images.png".format(opt.dataset, opt.attack_mode)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    if not epoch % 10:
        # Save figures (tfboard)
        tf_writer.add_scalars(
            "Accuracy/lambda_div_{}/".format(opt.lambda_div),
            {"Clean": acc_clean},
            epoch,
        )

        tf_writer.add_scalars("Loss/lambda_div_{}".format(opt.lambda_div), {"CE": loss_ce}, epoch)

    schedulerC.step()
    schedulerG.step()


def eval(
    netC,
    netG,
    netM,
    optimizerC,
    optimizerG,
    schedulerC,
    schedulerG,
    test_dl1,
    epoch,
    best_acc_clean,
    opt,
):
    netC.eval()
    netG.eval()
    print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    for batch_idx, (inputs1, targets1) in enumerate(test_dl1):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            bs = inputs1.shape[0]

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
    print(
        " Result: Best Clean Accuracy: {:.3f}| Clean Accuracy: {:.3f}".format(
            best_acc_clean, avg_acc_clean
        )
    )
    if avg_acc_clean > best_acc_clean:
        print(" Saving!!")
        best_acc_clean = avg_acc_clean
        state_dict = {
            "netC": netC.state_dict(),
            "netG": netG.state_dict(),
            "netM": netM.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "best_acc_clean": best_acc_clean,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
        torch.save(state_dict, ckpt_path)
    return best_acc_clean, epoch

def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    netG = Generator(opt).to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    netM = Generator(opt, out_channels=1).to(opt.device)
    optimizerM = torch.optim.Adam(netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tf_writer = SummaryWriter(log_dir=log_dir)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    ckpt_path_mask = os.path.join(ckpt_folder, "mask", "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        netG.load_state_dict(state_dict["netG"])
        netM.load_state_dict(state_dict["netM"])
        epoch = state_dict["epoch"] + 1
        optimizerC.load_state_dict(state_dict["optimizerC"])
        optimizerG.load_state_dict(state_dict["optimizerG"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        schedulerG.load_state_dict(state_dict["schedulerG"])
        best_acc_clean = state_dict["best_acc_clean"]
        opt = state_dict["opt"]
        print("Continue training")
    elif os.path.exists(ckpt_path_mask):
        state_dict = torch.load(ckpt_path_mask)
        netM.load_state_dict(state_dict["netM"])
        optimizerM.load_state_dict(state_dict["optimizerM"])
        schedulerM.load_state_dict(state_dict["schedulerM"])
        opt = state_dict["opt"]
        best_acc_clean = 0.0
        epoch = state_dict["epoch"] + 1
        print("Continue training ---")
    else:
        # Prepare mask
        best_acc_clean = 0.0
        epoch = 1

        # Reset tensorboard
        # shutil.rmtree(log_dir)
        # os.makedirs(log_dir)
        print("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)

    for i in range(opt.n_iters):
        print(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )
        train_step(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            train_dl1,
            epoch,
            opt,
            tf_writer,
        )

        best_acc_clean, epoch = eval(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            test_dl1,
            epoch,
            best_acc_clean,
            opt,
        )
        epoch += 1
        if epoch > opt.n_iters:
            break


def main():
    opt = config.get_arguments().parse_args()
    use_cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")
    train(opt)


if __name__ == "__main__":
    opt = config.get_arguments().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    main()

## python train.py --dataset cifar10 --target_label 0 --gpu 0
## python train.py --dataset gtsrb --target_label 2 --gpu 0
