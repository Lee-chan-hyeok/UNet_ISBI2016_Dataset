import os
import logging
import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import config as config
from unet import ChleeUNet
from dataset import ISICDataset
from transform import transforms
from loss import DiceLoss
from utils import dsc


def run_epoch(
        cfg: config,
        model: ChleeUNet,
        data_loader: dict,
        optimizer: torch.optim,
        loss_fn: DiceLoss,
        epoch: int,
        device,
    ):

    # ========================= train =========================
    model.train()
    train_loader = data_loader["train_loader"]
    
    train_epoch_loss = []
    train_epoch_dsc = []
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.item())

        # Dice score 계산
        batch_dsc = dsc(
            y_pred.detach().numpy(),
            y_true.detach().numpy(),
            False
        )

        train_epoch_dsc.append(batch_dsc)

    mean_train_epoch_loss = np.mean(train_epoch_loss)   # 평균 train loss 값
    mean_train_epoch_dsc = np.mean(train_epoch_dsc)
    print(f"Epoch {epoch+1} | Train Loss: {mean_train_epoch_loss:.4f}")

    # ========================= valid =========================
    model.eval()
    valid_loader = data_loader["valid_loader"]

    valid_epoch_loss = []
    valid_epoch_dsc = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch+1} Validate")):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            valid_epoch_loss.append(loss.item())

            # Dice score 계산
            batch_dsc = dsc(
                y_pred.detach().numpy(),
                y_true.detach().numpy(),
                False
            )

            valid_epoch_dsc.append(batch_dsc)

    mean_valid_epoch_loss = np.mean(valid_epoch_loss)
    mean_valid_epoch_dsc = np.mean(valid_epoch_dsc)
    print(f"Epoch {epoch+1} | Valid Loss: {mean_valid_epoch_loss:.4f} | Valid DSC: {mean_valid_epoch_dsc:.4f}")

    loss_dict = {
        "train_loss": mean_train_epoch_loss,
        "valid_loss": mean_valid_epoch_loss
    }

    dsc_dict = {
        "train_dsc": mean_train_epoch_dsc,
        "valid_dsc": mean_valid_epoch_dsc
    }

    return loss_dict, dsc_dict



def main():
    cfg = config
    model_save_path = os.path.join("checkpoint", cfg.save_path)
    os.makedirs(model_save_path, exist_ok=True)

    # ================ 로그 파일 + 화면 출력 설정 ================
    logging.basicConfig(
    filename=os.path.join(model_save_path, "train_log.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    # =========================================================


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"device: {device}")

    # model 초기화
    model = ChleeUNet(in_channel=3, hidden_channel=64, out_channel=1)
    model.to(device)
    # summary(model, (1, 3, 572, 572))

    # dataset 초기화
    train_dataset = ISICDataset(
        images_dir=cfg.images,
        transform=transforms(scale=cfg.aug_scale, angle=cfg.aug_angle, flip_prob=0.5),
        subset="train",
        validation_cases=cfg.validation_cases,
        seed=cfg.seed,
        sampling_fraction=cfg.sampling_fraction
    )
    valid_dataset = ISICDataset(
        images_dir=cfg.images,
        transform=None,
        subset="validation",
        validation_cases=cfg.validation_cases,
        seed=cfg.seed,
        sampling_fraction=1.0
    )

    def worker_init(worker_id):
        np.random.seed(cfg.seed + worker_id)

    # loader 초기화
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        worker_init_fn=worker_init
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.workers,
        worker_init_fn=worker_init
    )

    loader_dict = {
        "train_loader": train_loader,
        "valid_loader": valid_loader
    }

    # optimizer, loss 초기화
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
    loss_fn = DiceLoss()

    best_validation_dsc = 0.0
    
    patience = cfg.patience,
    early_stopping = cfg.early_stopping
    
    train_loss_list, valid_loss_list = [], []
    train_dsc_list, valid_dsc_list = [], []
    lr_list = []
    for epoch in range(cfg.epochs):
        loss_dict, dsc_dict = run_epoch(
            cfg,
            model,
            loader_dict,
            optimizer,
            loss_fn,
            epoch,
            device,
        )

        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        train_loss_list.append(loss_dict["train_loss"])
        valid_loss_list.append(loss_dict["valid_loss"])
        train_dsc_list.append(dsc_dict["train_dsc"])
        valid_dsc_list.append(dsc_dict["valid_dsc"])

        # early stopping
        if best_validation_dsc < dsc_dict["valid_dsc"]:
            best_validation_dsc = dsc_dict["valid_dsc"]
            patience=0
            torch.save(model.state_dict(), os.path.join(cfg.weights, "best.pt"))
            logging.info(f"----- Save best.pt model !!! -----")
        else:
            patience += 1
            if patience >= early_stopping:
                logging.info(f"----- Save best.pt model !!! -----")
                break

        








if __name__ == "__main__":
    main()