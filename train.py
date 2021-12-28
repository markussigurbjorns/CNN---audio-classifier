# -*- coding: utf-8 -*-
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from DatasetStruct import UniversalSoundDataset as usd
from CNNnet import CNNnet

#Annotiations file path
ANNOTATIONS = "Annotiations file path"
#Audio file patH
AUDIO_DIR = "Audio file path"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128

def create_dataloader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=128)
    return train_data_loader

def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        #calculate loss
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        
        #Backprog loss and uptade weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'loss{loss.item()}')
    
        
    
def train(model, data_loader, loss_function, optimizer, device, epochs):
    
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_function, optimizer, device)
        print("-------------------------")
    print("training is done")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64)

    usd_data = usd(ANNOTATIONS,AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device )

    train_data_loader = create_dataloader(usd_data, BATCH_SIZE)

    #build model
    CNNet = CNNnet().to(device)

    #instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNNet.parameters(), lr=.001)

    #train model
    train(CNNet, train_data_loader, loss_fn, optimizer, device, 10)

    #store model
    torch.save(CNNet.state_dict(), "CNNnet.pth")
    print("Model trained and stored at CNNnet.pth")



if __name__ == "__main__":
    main()