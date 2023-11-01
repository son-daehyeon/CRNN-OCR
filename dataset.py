import cv2
import torch
import albumentations


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, char2int):
        self.image_ids = list(data_file['images'])
        self.labels = list(data_file['labels'])
        self.char2int = char2int

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.aug = albumentations.Compose(
            [albumentations.Normalize(mean, std,
                                      max_pixel_value=255.0,
                                      always_apply=True)]
        )

        # Find the maximum label length in the dataset
        self.max_len = data_file['labels'].apply(lambda x: len(x)).max()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = cv2.imread(img_id)

        augmented = self.aug(image=img)

        img = augmented['image']
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        target = self.labels[idx]
        target = [self.char2int[i] for i in target]
        target_len = torch.LongTensor([len(target)])
        target += [0] * (self.max_len - len(target))
        target = torch.LongTensor(target)

        return img, target, target_len
