import os
import random
from pathlib import Path  # <--- [1] 수정: pathlib 임포트

import numpy as np
import torch
from PIL import Image  # <--- [2] 수정: PIL 임포트
from torch.utils.data import Dataset
import torchvision.transforms as T  # <--- [3] 수정: torchvision 임포트

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume
# (위 utils 함수들은 더 이상 필요하지 않습니다)


class ISICDataset(Dataset):  # <--- [4] 수정: 클래스명 변경
    """
    ISBI ISIC 2016 커스텀 데이터셋
    - Input: (3, 572, 572)
    - Output: (1, 388, 388)
    - __init__ 매개변수는 BrainSegmentationDataset과 호환되게 유지
    """

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,             # dataset root_dir
        transform=None,         # Augmentation
        image_size=(572, 572),         # (이 매개변수는 사용되지 않음, (572, 388)로 고정)
        subset="train",
        validation_cases=10,    # validation dataset 개수
        seed=42,
        sampling_fraction=1.0,  # train dataset sampling 비율
    ):
        assert subset in ["all", "train", "validation"]

        root_dir = Path(images_dir)
        self.image_dir = root_dir / "ISBI2016_ISIC_Part1_Training_Data"
        self.mask_dir = root_dir / "ISBI2016_ISIC_Part1_Training_GroundTruth"
        self.transform = transform

        # 1. 이미지 파일 경로
        all_image_files = sorted(list(self.image_dir.glob("*.jpg")))
        if not all_image_files:
            raise FileNotFoundError(f".jpg 이미지를 찾을 수 없습니다: {self.image_dir}")

        # 2. 이미지 파일 이름과 짝이 맞는 mask 찾기
        all_mask_files = []
        valid_image_files = []
        for img_path in all_image_files:
            stem = img_path.stem
            mask_name = f"{stem}_Segmentation.png"
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                valid_image_files.append(img_path)
                all_mask_files.append(mask_path)
            else:
                print(f"경고: {img_path.name}의 짝이 맞는 마스크가 없습니다. (건너뜀)")

        # 3. train/validation 분할 (BrainSegmentationDataset 로직 활용)
        if not subset == "all":
            random.seed(seed)
            indices = list(range(len(valid_image_files)))
            
            # validation_cases 수만큼 인덱스를 샘플링
            validation_indices = random.sample(indices, k=min(validation_cases, len(indices)))

            if subset == "validation":
                self.image_files = [valid_image_files[i] for i in validation_indices]
                self.mask_files = [all_mask_files[i] for i in validation_indices]
            else:  # "train" subset
                train_indices = sorted(
                    list(set(indices).difference(validation_indices))
                )

                # 4. <핵심> train 인덱스 리스트를 샘플링합니다.
                if sampling_fraction < 1.0:
                    num_train_files = len(train_indices)
                    num_samples = int(num_train_files * sampling_fraction)

                    print(f"--- [샘플링] 원본 학습 파일 수: {num_train_files}개")
                    print(f"--- [샘플링] {sampling_fraction*100:.0f}% 샘플링 적용: {num_samples}개 ---")

                    train_indices = random.sample(train_indices, k=num_samples)

                self.image_files = [valid_image_files[i] for i in train_indices]
                self.mask_files = [all_mask_files[i] for i in train_indices]
        else:
            self.image_files = valid_image_files
            self.mask_files = all_mask_files

        if not self.image_files:
             raise RuntimeError(f"'{subset}' 데이터셋에 사용할 파일이 없습니다.")

        print(f"'{subset}' 데이터셋 로드 완료. 파일 개수: {len(self.image_files)}개")

        # 5. 필요한 Transform들을 정의합니다. (Lazy Loading용)
        self.input_shape = image_size
        self.output_shape = (388, 388)

        # 이미지용: (572, 572)로 리사이즈
        self.resize_image = T.Resize(self.input_shape, interpolation=T.InterpolationMode.BICUBIC)
        
        # 마스크용: (572, 572)로 리사이즈 (정합)
        self.resize_mask = T.Resize(self.input_shape, interpolation=T.InterpolationMode.NEAREST)
        
        # 마스크 중앙을 (388, 388)로 잘라내기
        self.crop_mask = T.CenterCrop(self.output_shape)
        
        # 텐서로 변환
        self.to_tensor = T.ToTensor()

        # --- [6] 수정: Eager Loading (self.volumes=...) 및 슬라이스 인덱싱 로직 모두 제거 ---
        # (print("preprocessing..."), crop_sample, pad_sample 등 모두 제거)
        # (self.patient_slice_index, self.slice_weights 등 모두 제거)

    def __len__(self):
        # --- [7] 수정: 슬라이스 인덱스가 아닌 파일 리스트의 길이 반환 ---
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- [8] 수정: Lazy Loading 로직으로 덮어쓰기 ---
        
        # 1. 인덱스(idx)에 해당하는 파일 경로 가져오기
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 2. PIL을 사용하여 이미지와 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 'L': 8비트 흑백

        # 3. 이미지 전처리 (Input: 572x572)
        # 원본 -> (572, 572) 리사이즈
        image = self.resize_image(image)

        # 4. 마스크 전처리 (Ground Truth: 388x388)
        # 원본 -> (572, 572) 리사이즈
        mask = self.resize_mask(mask)

        # 5. 데이터 증강 (Augmentation) - (train.py에서 NumPy 기반 transform을 전달한다고 가정)
        # (만약 transform이 None이거나 torchvision transform이면 이 부분 수정 필요)
        if self.transform is not None:
            # PIL -> NumPy로 변환
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # transform.py의 (image, mask) 튜플 입력 방식 적용
            image_np, mask_np = self.transform((image_np, mask_np))
            
            # NumPy -> PIL로 다시 변환
            image = Image.fromarray(image_np.astype(np.uint8))
            mask = Image.fromarray(mask_np.astype(np.uint8))

        # 6. 텐서로 변환
        image = self.to_tensor(image) # (3, 572, 572)

        # (1, 572, 572)
        mask_tensor = self.to_tensor(mask) 
        
        # 7. 마스크 크롭 (1, 388, 388)
        mask_tensor = self.crop_mask(mask_tensor)

        # 8. 마스크 이진화 (0 또는 1)
        mask_tensor = (mask_tensor > 0.5).float()

        return image, mask_tensor