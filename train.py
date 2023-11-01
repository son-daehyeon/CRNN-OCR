import torch
import config
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from utils import load_obj
from network import ConvRNN
from dataset import MyDataset
from sklearn.model_selection import train_test_split

csv_path = config.preprocessing['csv_path']
char2int_path = config.preprocessing['char2int_path']
epochs = config.training['epochs']
batch_size = config.training['batch_size']
output_path = config.training['output_path']


def train(model, dataloader, criterion, device, optimizer=None, test=False):
    # Set the model to train or eval mode
    model.eval() if test else model.train()

    # Initialize loss list
    loss = []

    # Iterate over the data
    for inp, tgt, tgt_len in tqdm(dataloader):
        # Move the data to the device
        inp = inp.to(device)
        tgt = tgt.to(device)
        tgt_len = tgt_len.to(device)

        # Get the output from the model
        out = model(inp)
        out = out.permute(1, 0, 2)

        # Create input length tensor
        inp_len = torch.LongTensor([40] * out.shape[1])

        # Calculate the loss
        log_probs = nn.functional.log_softmax(out, 2)
        loss_ = criterion(log_probs, tgt, inp_len, tgt_len)

        # Backpropagate the loss and update the parameters
        if not test:
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        # Append the loss to the loss list
        loss.append(loss_.item())

    return np.mean(loss)


if __name__ == '__main__':
    # Load the data file
    data_file = pd.read_csv(csv_path)
    data_file.fillna('null', inplace=True)

    # Load the char2int dictionary
    char2int = load_obj(char2int_path)

    # Set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Split the data into train and validation sets
    train_file, valid_file = train_test_split(data_file, test_size=0.2)

    # Create the dataset objects
    train_dataset = MyDataset(train_file, char2int)
    valid_dataset = MyDataset(valid_file, char2int)

    # Define the loss function
    criterion = nn.CTCLoss(reduction='sum')
    criterion.to(device)

    # Get the number of classes
    model = ConvRNN(len(char2int))
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)

    # Initialize the best loss
    best_loss = 1e7

    for i in range(epochs):
        print(f'Epoch {i + 1} of {epochs}...')

        train_loss = train(model, train_loader, criterion, device, optimizer, test=False)
        valid_loss = train(model, valid_loader, criterion, device, test=True)

        print(f'Train Loss: {round(train_loss, 4)}, Valid Loss: {round(valid_loss, 4)}')

        if valid_loss < best_loss:
            print('Validation Loss improved, saving Model File...')
            torch.save(model.state_dict(), output_path)
            best_loss = valid_loss
