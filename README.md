# 24510115_shakespeare

파일 설명
dataset.py
이 파일에는 데이터 로딩 및 전처리를 위한 Shakespeare 클래스 가 들어있습니다.
해당 클래스는 파일 이름을 인자로 받고, 해당 이름을 가진 txt 파일  불러와 텍스트 파일에서 불러온 문자를 인덱스로 변환 후 길이 30의 시퀀스로 분할합니다.

model.py
이 파일에는 두 개의 클래스가 정의되어있음 :

CharRNN: vanilla RNN 을 기반으로 하는 모델
CharLSTM: LSTM 을 기반으로 하는 모델 .
두 클래스 모두 순전파 및 은닉 상태 초기화를 위한 메서드를 포함합니다.


main.py
이 파일은 훈련 및 검증 로직을 포함합니다:

train(): 모델을 훈련시키고 평균 훈련 손실을 계산합니다.
validate(): 모델을 검증하고 평균 검증 손실을 계산합니다.
main(): 데이터 로더, 모델, 손실 함수 및 옵티마이저를 설정한 후 훈련 및 검증 과정을 실행합니다.

실제 실험은 ipynb에서 진행을 했고, 히든 레이어 를 몇겹으로 쌓아야 가장 효율적인지 테스트 하기 위해서 일단 히든 레이어별 20epoch씩 실험을 한 후.
가장 성능이 좋았던 레이어를 골라서 다시 50 epoch 실험을 진행했습니다.

레이어 개수의 경우 주피터 노트북에서 테스트하고 main.py 를 만듬, 각 layer 별 실험 코드는  주피터 노트북 내에있습니다. 

각 레이어별 20 epoch 실험 plot
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/b1706b4b-de93-4b68-b0be-a238c138c31b)
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/8c1e0cfc-e019-482d-b18b-75f777b284ae)
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/3149ec00-d181-488e-aa02-be5d132c912b)
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/7c496d0e-e19f-41a4-b88d-fccfcc052b84)

가장 좋은 결과를 보인 히든 레이어수로 다시 테스트 돌린 plot들 

RNN 50 epoch 
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/8cc31ddd-fb91-419c-b42e-644d61400069)

Training with 6 hidden layers

Epoch 1/50, RNN Loss: 1.4033, RNN Accuracy: 0.5729 

Epoch 2/50, RNN Loss: 1.2924, RNN Accuracy: 0.6006

Epoch 3/50, RNN Loss: 1.2190, RNN Accuracy: 0.6215

Epoch 4/50, RNN Loss: 1.1753, RNN Accuracy: 0.6337
Epoch 5/50, RNN Loss: 1.1453, RNN Accuracy: 0.6435
...
Epoch 45/50, RNN Loss: 1.0229, RNN Accuracy: 0.6795
Epoch 46/50, RNN Loss: 1.0159, RNN Accuracy: 0.6831
Epoch 47/50, RNN Loss: 1.0208, RNN Accuracy: 0.6814
Epoch 48/50, RNN Loss: 1.0179, RNN Accuracy: 0.6816
Epoch 49/50, RNN Loss: 1.0173, RNN Accuracy: 0.6823
Epoch 50/50, RNN Loss: 1.0231, RNN Accuracy: 0.6798


LSTM 50 epoch 
![image](https://github.com/hansanghooon/24510115_shakespeare/assets/132417290/e39ba979-6bb6-407d-a75c-1f5c11e2a0c2)

Training with 5 hidden layers

Epoch 1/50,LSTM Loss: 1.5324, LSTM Accuracy: 0.5404
Epoch 2/50,LSTM Loss: 1.3236, LSTM Accuracy: 0.5936
Epoch 3/50,LSTM Loss: 1.2043, LSTM Accuracy: 0.6255
Epoch 4/50,LSTM Loss: 1.0868, LSTM Accuracy: 0.6627
Epoch 5/50,LSTM Loss: 1.0114, LSTM Accuracy: 0.6856
...
Epoch 45/50,LSTM Loss: 0.6142, LSTM Accuracy: 0.8189
Epoch 46/50,LSTM Loss: 0.6094, LSTM Accuracy: 0.8207
Epoch 47/50,LSTM Loss: 0.6080, LSTM Accuracy: 0.8210
Epoch 48/50,LSTM Loss: 0.6122, LSTM Accuracy: 0.8197
Epoch 49/50,LSTM Loss: 0.6064, LSTM Accuracy: 0.8219
Epoch 50/50,LSTM Loss: 0.6061, LSTM Accuracy: 0.8209

genrate.py
high temperatures (T>1):

높 값은 확률 분포를 부드럽게 하여 더 균일하게 만듭니다. 이는 모델이 덜 확률적인 문자도 샘플링할 가능성이 높아져, 더 다양한 텍스트를 생성하지만덜 일관된 결과를 초래할 가능성이 존재 합니다.


low temperatures (0<𝑇<1):
모델의 예측값에 훨씬더 의존된 결과를 가집니다(새로운 결과가 나오는것이 아니라 확률이 높은 기존의 결과를 선택할 가능성이 높아짐) 이는 예측가능하고 일관된 텍스트를 형성하게 됩니다.



T=1.0인 경우, 생성된 텍스트는 일관성과 다양성의 균형을 유지하여 일반적으로 그럴듯한 결과를 만듭니다.
