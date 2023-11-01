import cv2
import torch
import albumentations
import config

from utils import load_obj
from network import ConvRNN

if __name__ == '__main__':
    test_img = config.project_dir + '/data/images/00036806.jpg'
    int2char_path = config.preprocessing['int2char_path']
    output_path = config.training['output_path']

    int2char = load_obj(int2char_path)
    n_classes = len(int2char)

    # Initialize the model
    model = ConvRNN(n_classes)

    # Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(output_path, map_location=device))

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    # Define the mean and standard deviation of the dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Read the image
    img = cv2.imread(test_img)

    img_aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )

    augmented = img_aug(image=img)
    img = augmented['image']
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)

    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    out = model(img)
    out = torch.squeeze(out, 0)
    out = out.softmax(1)

    pred = torch.argmax(out, 1)
    pred = pred.tolist()
    int2char[0] = 'ph'

    out = [int2char[i] for i in pred]

    res = list()
    res.append(out[0])
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            res.append(out[i])

    res = [i for i in res if i != 'ph']

    print('input: ' + test_img[-25:])
    print('output: ' + ''.join(res))
