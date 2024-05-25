import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import time

from dataset import Shakespeare
from model import CharRNN, CharLSTM


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 2)
    correct = (predicted == targets).float()
    accuracy = correct.sum() / targets.numel()
    return accuracy.item()



def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        accuracy = calculate_accuracy(outputs, targets)
        total_accuracy += accuracy
    return total_loss / len(trn_loader), total_accuracy / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, targets)
            total_accuracy += accuracy
    return total_loss / len(val_loader), total_accuracy / len(val_loader)





def main():
    dataset = Shakespeare('shakespeare.txt')
    batch_size = 64
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 24

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    vocab_size = len(dataset.chars)
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50

    start=time.time()
    print(f'\nTraining with {5} hidden layers\n')

    model_lstm = CharLSTM(vocab_size, hidden_size, 5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_lstm = optim.RMSprop(model_lstm.parameters(), lr=0.001)


    trn_losses_rnn, val_losses_rnn = [], []
    trn_acc_rnn, val_acc_rnn = [], []
    trn_losses_lstm = []
    val_losses_lstm = []
    trn_acc_lstm = []
    val_acc_lstm = []
    num_layers=5
    for epoch in range(num_epochs):

        trn_loss_lstm, trn_accuracy_lstm = train(model_lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss_lstm, val_accuracy_lstm = validate(model_lstm, val_loader, device, criterion)
        trn_losses_lstm.append(trn_loss_lstm)
        val_losses_lstm.append(val_loss_lstm)
        trn_acc_lstm.append(trn_accuracy_lstm)
        val_acc_lstm.append(val_accuracy_lstm)

        print(f'Epoch {epoch+1}/{num_epochs}'
                f'LSTM Loss: {val_loss_lstm:.4f}, LSTM Accuracy: {val_accuracy_lstm:.4f}')

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(trn_losses_lstm, label='Train Loss LSTM')
    plt.plot(val_losses_lstm, label='Validation Loss LSTM')
    plt.title(f'Training and Validation Loss with {num_layers} Hidden Layers')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(trn_acc_lstm, label='Train Accuracy LSTM')
    plt.plot(val_acc_lstm, label='Validation Accuracy LSTM')
    plt.title(f'Training and Validation Accuracy with {num_layers} Hidden Layers')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print(time.time()-start)

    # model_rnn과 model_lstm은 이미 학습된 모델 객체라고 가정합니다.
    # num_layers는 히든 레이어 수를 나타냅니다.

    # 모델 파라미터를 저장할 파일 이름 지정
    lstm_model_path = f'char_lstm_{5}_layers.pth'

    # 모델 파라미터 저장
    torch.save(model_lstm.state_dict(), lstm_model_path)

    print(f"Saved LSTM model with {5} hidden layers to {lstm_model_path}")

    dataset = Shakespeare('shakespeare.txt')
    batch_size = 64
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 24

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    vocab_size = len(dataset.chars)
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50
    num_layers=6

    start=time.time()
    print(f'\nTraining with {num_layers} hidden layers\n')

    model_rnn = CharRNN(vocab_size, hidden_size, num_layers).to(device)
    model_lstm = CharLSTM(vocab_size, hidden_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

    trn_losses_rnn, val_losses_rnn = [], []
    trn_acc_rnn, val_acc_rnn = [], []
    trn_losses_lstm = []
    val_losses_lstm = []
    trn_acc_lstm = []
    val_acc_lstm = []

    for epoch in range(num_epochs):
        trn_loss_rnn, trn_accuracy_rnn = train(model_rnn, train_loader, device, criterion, optimizer_rnn)
        val_loss_rnn, val_accuracy_rnn = validate(model_rnn, val_loader, device, criterion)
        trn_losses_rnn.append(trn_loss_rnn)
        val_losses_rnn.append(val_loss_rnn)
        trn_acc_rnn.append(trn_accuracy_rnn)
        val_acc_rnn.append(val_accuracy_rnn)

        trn_loss_lstm, trn_accuracy_lstm = train(model_lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss_lstm, val_accuracy_lstm = validate(model_lstm, val_loader, device, criterion)
        trn_losses_lstm.append(trn_loss_lstm)
        val_losses_lstm.append(val_loss_lstm)
        trn_acc_lstm.append(trn_accuracy_lstm)
        val_acc_lstm.append(val_accuracy_lstm)

        print(f'Epoch {epoch+1}/{num_epochs}, RNN Loss: {val_loss_rnn:.4f}, RNN Accuracy: {val_accuracy_rnn:.4f}, '
               )

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(trn_losses_rnn, label='Train Loss RNN')
    plt.plot(val_losses_rnn, label='Validation Loss RNN')

    plt.title(f'Training and Validation Loss with {num_layers} Hidden Layers')
    plt.legend() 
    plt.subplot(2, 1, 2)
    plt.plot(trn_acc_rnn, label='Train Accuracy RNN')
    plt.plot(val_acc_rnn, label='Validation Accuracy RNN')
    plt.title(f'Training and Validation Accuracy with {num_layers} Hidden Layers')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    print(time.time()-start)


if __name__ == '__main__':
    main()