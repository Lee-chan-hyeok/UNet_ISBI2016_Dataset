from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import random
import torch


class ISICDataset(Dataset):
    def __init__(
        self,
        images_dir,
        image_size=(572, 572),
        subset="train",
        validation_ratio=0.1,
        seed=42,
    ):
        assert subset in ["train", "valid"]

        root_dir = Path(images_dir)
        image_dir = root_dir / "ISBI2016_ISIC_Part1_Training_Data"
        mask_dir = root_dir / "ISBI2016_ISIC_Part1_Training_GroundTruth"

        all_image_files = sorted(list(image_dir.glob("*.jpg")))
        if not all_image_files:
            raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_dir}")

        all_pairs = []
        for img_path in all_image_files:
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}_Segmentation.png"
            all_pairs.append((img_path, mask_path))

        random.seed(seed)
        random.shuffle(all_pairs)

        # train, valid 분할
        split_idx = int(len(all_pairs) * (1 - validation_ratio))
        if subset == "train":
            self.files = all_pairs[:split_idx]
        else:
            self.files = all_pairs[split_idx:]

        print(f"{subset} 데이터셋 로드 완료 ({len(self.files)}개)")

        self.resize_img = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.resize_mask = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)
        self.center_crop = transforms.CenterCrop((388, 388))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # resize
        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        # to tensor
        image = self.to_tensor(image)  # (3, 572, 572)
        mask = self.to_tensor(mask)    # (1, 572, 572)

        # center crop
        mask = self.center_crop(mask)
        mask = (mask > 0.5).float()    # 0 또는 1로 변환

        return image, mask
