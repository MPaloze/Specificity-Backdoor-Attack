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
from torchvision import transforms
from random_color import white_f
from PIL import Image
from random_color import  random_color_f
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def add_white_trigger(img):
    img = transforms.ToPILImage()(img)
    img.paste(white_f(),(24,24))
    #img.show()
    return img

def add_various_trigger(img):
    img = transforms.ToPILImage()(img)
    img.paste(random_color_f(), (24, 24))
    # img.show()
    return img

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(inputs, targets, opt, num):
    bd_targets = create_targets_bd(targets, opt)
    for i in range(num):
        img_p = add_white_trigger(inputs[i])
        #img_p.show()
        t = transforms.ToTensor()(img_p)
        t = t.unsqueeze(0)
        #print(t.shape)
        if i == 0:
            bd_inputs = t
        else:
            bd_inputs = torch.cat([bd_inputs, t], 0)

    return bd_inputs, bd_targets

def create_cover(inputs, targets, opt, num):
    #bd_targets = create_targets_bd(targets, opt)
    for j in range(num):
        img_p = add_various_trigger(inputs[j])
        #img_p.show()
        t = transforms.ToTensor()(img_p)
        t = t.unsqueeze(0)
        #print(t.shape)
        if j == 0:
            bd_inputs = t
        else:
            bd_inputs = torch.cat([bd_inputs, t], 0)

    return bd_inputs, targets

def train_step(
    netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, epoch, opt, tf_writer
):
    netC.train()
    netG.train()
    print(" Training:")
    total = 0
    total_bd = 0
    total_clean = 0

    total_correct_clean = 0
    total_bd_correct = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1) in enumerate(train_dl1):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(opt.p_attack * bs)*2


        inputs_bd, targets_bd = create_bd(inputs1[:num_bd//2], targets1[:num_bd//2], opt, num_bd//2)
        inputs_cover, targets_cover = create_cover(inputs1[num_bd//2:num_bd], targets1[num_bd//2:num_bd], opt, num_bd//2)

        inputs_bd = inputs_bd.to(opt.device)
        inputs_cover = inputs_cover.to(opt.device)

        debug_inputs = torch.cat((inputs_bd, inputs_cover),0)

        total_inputs = torch.cat((inputs_bd, inputs_cover, inputs1[num_bd :]), 0)
        total_targets = torch.cat((targets_bd, targets_cover, targets1[num_bd:]), 0)

        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        total_loss = loss_ce
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

        total += bs
        total_bd += (num_bd//2)
        total_clean += bs - (num_bd//2)

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd//2:], dim=1) == total_targets[num_bd//2:]
        )

        total_bd_correct += torch.sum(torch.argmax(preds[:num_bd//2], dim=1) == targets_bd)
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = total_bd_correct * 100.0 / total_bd
        if not batch_idx % 50:
            print(batch_idx, len(train_dl1), "CE loss: {:.4f} - Clean Accuracy: {:.3f} | BD Accuracy: {:.3f}".format(
                avg_loss, acc_clean, acc_bd
            ))

        # Saving images for debugging

        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], debug_inputs), dim=2))
            file_name = "{}_{}_images.png".format(opt.dataset, opt.attack_mode)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    if not epoch % 10:
        # Save figures (tfboard)
        tf_writer.add_scalars(
            "Accuracy/lambda_div_{}/".format(opt.lambda_div),
            {"Clean": acc_clean, "BD": acc_bd},
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
    best_acc_bd,
    opt,
):
    netC.eval()
    netG.eval()
    print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_bd = 0.0
    for batch_idx, (inputs1, targets1) in enumerate(test_dl1):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            bs = inputs1.shape[0]

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean

            inputs_bd, targets_bd = create_bd(inputs1, targets1, opt, bs)
            inputs_bd = inputs_bd.to(opt.device)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total

    print(
        " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f}| Clean Accuracy: {:.3f} - Backdoor Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd,  avg_acc_clean, avg_acc_bd
        )
    )
    if avg_acc_clean + avg_acc_bd > best_acc_clean + best_acc_bd:
        print(" Saving!!")
        best_acc_clean = avg_acc_clean
        best_acc_bd = avg_acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "netG": netG.state_dict(),
            "netM": netM.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "best_acc_clean": best_acc_clean,
            "best_acc_bd": best_acc_bd,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_'+str(opt.target_label))
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
        torch.save(state_dict, ckpt_path)
    return best_acc_clean, best_acc_bd, epoch

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
        best_acc_bd = state_dict["best_acc_bd"]
        opt = state_dict["opt"]
        print("Continue training")
    elif os.path.exists(ckpt_path_mask):
        state_dict = torch.load(ckpt_path_mask)
        netM.load_state_dict(state_dict["netM"])
        optimizerM.load_state_dict(state_dict["optimizerM"])
        schedulerM.load_state_dict(state_dict["schedulerM"])
        opt = state_dict["opt"]
        best_acc_clean = 0.0
        best_acc_bd = 0.0
        epoch = state_dict["epoch"] + 1
        print("Continue training ---")
    else:
        # Prepare mask
        best_acc_clean = 0.0
        best_acc_bd = 0.0
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

        best_acc_clean, best_acc_bd, epoch = eval(
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
            best_acc_bd,
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
