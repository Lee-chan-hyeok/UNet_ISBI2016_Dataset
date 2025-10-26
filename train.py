import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- [1] 제공된 3개 파일 + 필수 파일 임포트 ---
from unet import ChleeUNet as UNet          # unet.py (제공됨)
from dataset import ISICDataset as Dataset  # dataset.py (제공됨, ISICDataset으로 수정됨)
from utils import dsc, log_images           # utils.py (제공됨, dsc와 log_images 사용)

# --- [!] 필수 가정 파일 ---
# (원본 GitHub에 포함된 파일들)
from loss import DiceLoss
from transform import transforms

def main(args):
    # 1. 폴더 생성 및 설정 저장
    makedirs(args)
    snapshotargs(args)

    # 2. 장치 설정
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    print(f"--- 학습 장치: {device} ---")
    
    # 3. 데이터 로더 준비
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    # 4. 모델, 손실 함수, 옵티마이저 초기화
    unet = UNet(
        in_channel=Dataset.in_channels, 
        out_channel=Dataset.out_channels
    )
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    # 5. 로거(TensorBoard) 초기화
    logger = SummaryWriter(log_dir=args.logs)
    print(f"--- TensorBoard 로그 경로: {args.logs} ---")

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs, desc="Epoch"):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            epoch_loss = []
            epoch_dsc = []

            for i, data in enumerate(tqdm(loaders[phase], desc=f"{phase} Phase", leave=False)):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # y_pred shape: (B, 1, 388, 388)
                    y_pred = unet(x) 
                    
                    # loss 계산 (DiceLoss)
                    loss = dsc_loss(y_pred, y_true)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        epoch_loss.append(loss.item())
                    
                    if phase == "valid":
                        epoch_loss.append(loss.item())
                        
                        # [3] metric 계산 (utils.py의 dsc 함수 사용)
                        # lcc=False (학습 초기에 예측값이 비어있는 경우 에러 방지)
                        batch_dsc = dsc(
                            y_pred.detach().cpu().numpy(), 
                            y_true.detach().cpu().numpy(),
                            lcc=False 
                        )
                        epoch_dsc.append(batch_dsc)
                        
                        # 검증 이미지 로깅
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = f"epoch_{epoch}_batch_{i}"
                                num_images = min(args.vis_images - i * args.batch_size, args.batch_size)
                                
                                images_to_log = log_images(x, y_true, y_pred, channel=0)[:num_images] # channel 0 (R)
                                for idx, img in enumerate(images_to_log):
                                    logger.add_image(f"{tag}/{idx}", img, global_step=step, dataformats='HWC')

            # --- 에포크 종료 후 로그 기록 ---
            mean_epoch_loss = np.mean(epoch_loss)
            logger.add_scalar(f"Loss/{phase}", mean_epoch_loss, step)
            
            if phase == "valid":
                mean_epoch_dsc = np.mean(epoch_dsc)
                logger.add_scalar("DSC/valid", mean_epoch_dsc, step)
                print(f"Epoch {epoch} | Valid Loss: {mean_epoch_loss:.4f} | Valid DSC: {mean_epoch_dsc:.4f}")

                # 베스트 모델 저장
                if mean_epoch_dsc > best_validation_dsc:
                    best_validation_dsc = mean_epoch_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet_best.pt"))
                    print(f"--- 새 베스트 모델 저장 (DSC: {best_validation_dsc:.4f}) ---")
            
            if phase == "train":
                 print(f"Epoch {epoch} | Train Loss: {mean_epoch_loss:.4f}")

    print(f"--- 학습 종료 ---")
    print(f"Best validation mean DSC: {best_validation_dsc:.4f}")
    logger.close()


def data_loaders(args):
    """데이터 로더 생성"""
    
    # [4] dataset.py (ISICDataset) 초기화
    # transform.py에서 transforms 함수를 가져와 데이터 증강 적용
    dataset_train = Dataset(
        images_dir=args.images, # ISIC 데이터셋의 루트 폴더
        subset="train",
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
        validation_cases=args.validation_cases,
        seed=args.seed,
        sampling_fraction=args.sampling_fraction
    )

    dataset_valid = Dataset(
        images_dir=args.images,
        subset="validation",
        transform=None, # 검증 데이터셋에는 증강 미적용
        validation_cases=args.validation_cases,
        seed=args.seed,
        sampling_fraction=1.0 # 검증 데이터셋은 항상 100% 사용
    )

    def worker_init(worker_id):
        np.random.seed(args.seed + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

# --- 유틸리티 함수들 ---

def makedirs(args):
    """폴더 생성"""
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    """실행 설정(args)을 json 파일로 저장"""
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4)

# --- [5] 메인 실행 (argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="U-Net (padding=0) Training for ISIC Dataset"
    )

    # --- 학습 파라미터 ---
    parser.add_argument(
        "--epochs", type=int, default=100, help="학습 에포크 수"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="배치 사이즈 (572x572는 크므로 작게 설정)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning Rate"
    )
    
    # --- 장치 및 경로 ---
    parser.add_argument(
        "--device", type=str, default="cpu", help="학습 장치 (e.g., 'cuda:0' or 'cpu')"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="DataLoader 워커 수 (CPU 사용 시 0 권장)"
    )
    parser.add_argument(
        "--images", type=str, default="./data", help="data 데이터셋 루트 폴더"
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="모델 가중치 저장 폴더"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="TensorBoard 로그 저장 폴더"
    )

    # --- 데이터셋 파라미터 (dataset.py와 연동) ---
    parser.add_argument(
        "--validation-cases", type=int, default=10, help="검증용 데이터 개수"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="데이터 분할 및 worker 시드"
    )
    parser.add_argument(
        "--sampling-fraction", type=float, default=1.0, help="학습 데이터 샘플링 비율 (1.0 = 100%)"
    )

    # --- 데이터 증강 파라미터 (transform.py와 연동) ---
    parser.add_argument(
        "--aug-scale", type=float, default=0.05, help="Augmentation 스케일"
    )
    parser.add_argument(
        "--aug-angle", type=int, default=15, help="Augmentation 회전 각도"
    )
    
    # --- 로깅 파라미터 ---
    parser.add_argument(
        "--vis-images", type=int, default=10, help="TensorBoard에 저장할 최대 검증 이미지 수"
    )
    parser.add_argument(
        "--vis-freq", type=int, default=5, help="N 에포크마다 검증 이미지 저장"
    )

    args = parser.parse_args()
    main(args)