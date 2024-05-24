import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, input_file):
        # 1. 텍스트 파일 읽기
        with open(input_file, 'r') as f:
            self.text = f.read()
        
        # 2. 고유 문자 추출 및 인덱스 사전 생성
        self.chars = sorted(set(self.text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # 3. 문자열을 인덱스 리스트로 변환
        self.data = [self.char_to_idx[char] for char in self.text]

        # 시퀀스 길이 설정 및 샘플 수 계산
        self.seq_length = 30
        self.num_samples = len(self.data) - self.seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 4. 시퀀스 생성
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        
        # 5. 텐서로 변환하여 반환
        return torch.tensor(input_seq), torch.tensor(target_seq)

if __name__ == '__main__':
    dataset = Shakespeare('shakespeare.txt')
    print(f'Dataset size: {len(dataset)}')
    print(f'First item: {dataset[0]}')
