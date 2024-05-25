# 24510115_shakespeare

파일 설명
dataset.py
이 파일에는 데이터 로딩 및 전처리를 위한 Shakespeare 클래스가 포함되어 있습니다.
해당 클래스는 파일 이름을 인자로 받아서, 해당 이름을 가진 txt 파일  불러옵니다.
텍스트 파일에서 불러온 문자를 인덱스로 변환 후 길이 30의 시퀀스로 분할합니다.

model.py
이 파일에는 두 개의 클래스가 정의되어 있습니다:

CharRNN: 문자 단위 모델링을 위한 vanilla RNN을 구현합니다.
CharLSTM: 문자 단위 모델링을 위한 LSTM을 구현합니다.
두 클래스 모두 순전파 및 은닉 상태 초기화를 위한 메서드를 포함합니다.


main.py
이 파일은 훈련 및 검증 로직을 포함합니다:

train(): 모델을 훈련시키고 평균 훈련 손실을 계산합니다.
validate(): 모델을 검증하고 평균 검증 손실을 계산합니다.
main(): 데이터 로더, 모델, 손실 함수 및 옵티마이저를 설정한 후 훈련 및 검증 과정을 실행합니다.

실제 실험은 ipynb에서 진행을 했고, 히든 레이어 를 몇겹으로 쌓아야 가장 효율적인지 테스트 하기 위해서 일단 히든 레이어별 20epoch씩 실험을 한 후.
가장 성능이 좋았던 레이어를 골라서 다시 50 epoch 실험을 진행. 
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/b2ffa9ae-e6e0-4ec2-a6ac-db260ade5baf)

![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/e39ba979-6bb6-407d-a75c-1f5c11e2a0c2)

