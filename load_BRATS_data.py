import torch
import random
import numpy as np
from pathlib import Path
from einops import repeat
from scipy.ndimage import zoom
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    # Randomly flip the image and label horizontally
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=(1, 2))
        label = np.flip(label, axis=1)

    # Randomly flip the image and label vertically
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=(0, 2))
        label = np.flip(label, axis=0)

    # Randomly rotate the image and label by 0, 90, 180, or 270 degrees
    rots = np.random.randint(0, 4)
    image = np.rot90(image, rots, axes=(1, 2))
    label = np.rot90(label, rots, axes=(0, 1))

    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res, test=False):
        self.output_size = output_size
        self.low_res = low_res
        self.test = test

    def __call__(self, sample):
        image, label, label_four = sample['image'], sample['label'], sample['label_four']
        label_four = np.stack(label_four, axis=0).astype(np.int64)
        label = label.squeeze()

        if not self.test:
            image, label = random_rot_flip(image, label)

        image_oc = image.copy()
        x, y = image.shape[-2:]

        # Resize image and label to the specified output size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)

        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))

        sample = {
            'image': image,
            'label': label.long(),
            'low_res_label': low_res_label.long(),
            'image_oc': image_oc,
            'label_four': label_four
        }
        return sample

class BRATS2017(Dataset):
    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        self.samples = []

        root = Path(dataset_location)

        # Collect .npz files from all splits present under root
        for split in ['train', 'val', 'test']:
            split_dir = root / split
            if split_dir.exists():
                self.samples.extend(sorted(split_dir.glob('*.npz')))

        # Fallback: user pointed directly at a folder of .npz files
        if not self.samples:
            self.samples = sorted(root.glob('*.npz'))

        if not self.samples:
            raise FileNotFoundError(
                f"No .npz files found under '{dataset_location}'. "
                "Please run prepare_brats2017.py first."
            )

        print(f"Loaded {len(self.samples)} slices from {dataset_location}")

    def __getitem__(self, index):
        data   = np.load(str(self.samples[index]))
        image  = data['image']   # float32 [H, W], values 0-1
        labels = data['label']   # uint8   [3, H, W], values 0 or 1

        # Add channel dim: [1, H, W]
        image = np.expand_dims(image, axis=0)

        # Randomly select one of the three ambiguous labels
        label_idx = random.randint(0, 2)
        label = labels[label_idx].astype(float)

        # Ensure the selected label is non-empty
        attempts = 0
        while label.sum() == 0 and attempts < 10:
            label_idx = random.randint(0, 2)
            label = labels[label_idx].astype(float)
            attempts += 1

        # label_four: all three ambiguous labels
        label_four = [labels[i].astype(float) for i in range(3)]

        # Convert to tensors
        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor).unsqueeze(0)

        image = np.array(image)
        label = np.array(label)

        sample = {'image': image, 'label': label, 'label_four': label_four}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)