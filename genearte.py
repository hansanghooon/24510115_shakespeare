# import torch
# import numpy as np
# from model import CharRNN, CharLSTM

# def generate(model, start_str, char_to_idx, idx_to_char, device, length=100, temperature=1.0):
#     model.eval()
#     input_seq = torch.tensor([char_to_idx[char] for char in start_str], dtype=torch.long).unsqueeze(0).to(device)
#     hidden = model.init_hidden(1)
#     if isinstance(hidden, tuple):
#         hidden = tuple(h.to(device) for h in hidden)
#     else:
#         hidden = hidden.to(device)
    
#     generated_str = start_str

#     with torch.no_grad():
#         for _ in range(length):
#             output, hidden = model(input_seq, hidden)
#             output = output.squeeze(0) / temperature
#             probabilities = torch.softmax(output[-1], dim=0).cpu().numpy()
#             next_char_idx = np.random.choice(len(probabilities), p=probabilities)
#             next_char = idx_to_char[next_char_idx]
#             generated_str += next_char
#             input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
#     return generated_str

# if __name__ == '__main__':
#     vocab_size = 6 
#     hidden_size = 128
#     num_layers_rnn = 6
#     num_layers_lstm = 5  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Shakespeare 데이터셋을 사용하여 사전 생성
#     with open('shakespeare.txt', 'r') as f:
#         text = f.read()

#     chars = sorted(set(text))
#     char_to_idx = {char: idx for idx, char in enumerate(chars)}
#     idx_to_char = {idx: char for char, idx in char_to_idx.items()}

#     start_str = "H"
#     length = 100

#     # RNN 모델 로드
#     model_rnn = CharRNN(vocab_size, hidden_size, num_layers_rnn).to(device)
#     model_rnn.load_state_dict(torch.load(f'char_rnn_{num_layers_rnn}_layers.pth'))

#     # LSTM 모델 로드
#     model_lstm = CharLSTM(vocab_size, hidden_size, num_layers_lstm).to(device)
#     model_lstm.load_state_dict(torch.load(f'char_lstm_{num_layers_lstm}_layers.pth'))

#     for temp in [0.5, 1.0, 1.5, 2.0]:
#         print(f"Temperature: {temp}")
#         generated_str_rnn = generate(model_rnn, start_str, char_to_idx, idx_to_char, device, length, temp)
#         generated_str_lstm = generate(model_lstm, start_str, char_to_idx, idx_to_char, device, length, temp)
#         print(f"RNN Generated Text: {generated_str_rnn}")
#         print(f"LSTM Generated Text: {generated_str_lstm}")
#         print("\n")
import torch
import numpy as np
from model import CharRNN, CharLSTM

def generate(model, start_str, char_to_idx, idx_to_char, device, length=100, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([char_to_idx[char] for char in start_str], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)
    
    generated_str = start_str

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output = output.squeeze(0) / temperature
            probabilities = torch.softmax(output[-1], dim=0).cpu().numpy()
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)
            next_char = idx_to_char[next_char_idx]
            generated_str += next_char
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated_str

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shakespeare 데이터셋을 사용하여 사전 생성
    with open('shakespeare.txt', 'r') as f:
        text = f.read()

    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = 62  # 저장된 모델과 일치하도록 설정

    start_str = "H"
    length = 100

    # RNN 모델 로드 (저장된 모델의 구조와 동일하게 설정)
    num_layers_rnn = 6  # 저장된 모델과 동일한 레이어 수로 설정
    hidden_size_rnn = 128  # 저장된 모델과 동일한 hidden_size로 설정
    model_rnn = CharRNN(vocab_size, hidden_size_rnn, num_layers=num_layers_rnn).to(device)
    model_rnn.load_state_dict(torch.load('/mnt/data/char_rnn_6_layers.pth', map_location=device))

    # LSTM 모델 로드 (저장된 모델의 구조와 동일하게 설정)
    num_layers_lstm = 5  # 저장된 모델과 동일한 레이어 수로 설정
    hidden_size_lstm = 128  # 저장된 모델과 동일한 hidden_size로 설정
    model_lstm = CharLSTM(vocab_size, hidden_size_lstm, num_layers=num_layers_lstm).to(device)
    model_lstm.load_state_dict(torch.load('/mnt/data/char_lstm_5_layers.pth', map_location=device))

    for temp in [0.5, 1.0, 1.5, 2.0]:
        print(f"Temperature: {temp}")
        generated_str_rnn = generate(model_rnn, start_str, char_to_idx, idx_to_char, device, length, temp)
        generated_str_lstm = generate(model_lstm, start_str, char_to_idx, idx_to_char, device, length, temp)
        print(f"RNN Generated Text: {generated_str_rnn}")
        print(f"LSTM Generated Text: {generated_str_lstm}")
        print("\n")
