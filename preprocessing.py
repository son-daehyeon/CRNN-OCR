import os
import config
import pandas as pd

from utils import save_obj

img_path = config.data['img_path']
label_path = config.data['label_path']
char2int_path = config.preprocessing['char2int_path']
int2char_path = config.preprocessing['int2char_path']
csv_path = config.preprocessing['csv_path']

if __name__ == '__main__':
    # Read the labels file
    labels = pd.read_table(label_path, header=None)
    labels.fillna('null', inplace=True)

    # Get the image files
    image_files = os.listdir(img_path)
    image_files.sort()
    image_files = [os.path.join(img_path, i) for i in image_files]

    # Get the unique characters
    unique_chars = list({l for word in labels[0] for l in word})
    unique_chars.sort()

    # Create a dictionary mapping characters to integers
    char2int = {a: i + 1 for i, a in enumerate(unique_chars)}
    int2char = {i + 1: a for i, a in enumerate(unique_chars)}

    save_obj(char2int, char2int_path)
    save_obj(int2char, int2char_path)

    # Create a dataframe for the image paths and labels
    data_file = pd.DataFrame({'images': image_files, 'labels': labels[0]})
    data_file.to_csv(csv_path, index=False)
