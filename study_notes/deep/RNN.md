## 순서가 있는 데이터: 순환 신경망 (RNN) ✍️📈

**1. MLP의 한계와 RNN의 등장**

- MLP(교안 5)는 모든 입력을 **'서로 독립적'**이라고 가정합니다.
    
- 예를 들어, "I am a boy"라는 문장을 MLP에 넣으면, "I", "am", "a", "boy"가 서로 아무 관계가 없는 4개의 데이터라고 봅니다.
    
- 하지만 **순서**가 바뀌면 "boy am I a"처럼 의미가 완전히 달라지죠.
    
- MLP는 이 '순서' 정보를 학습할 수 없습니다. 텍스트, 주가, 날씨 데이터 등 **순차적인(Sequential)** 데이터에 부적합합니다.
    

**2. RNN의 핵심 아이디어: '기억' (Memory)**

- RNN의 아이디어는 간단합니다. "이전 단계의 정보를 '기억'해서 다음 단계의 계산에 함께 사용하자."
    
- 이 '기억'을 전문 용어로 **은닉 상태(Hidden State, $h_t$)**라고 부릅니다.
    
- RNN은 마치 자기 자신에게로 돌아오는 루프(Loop)를 가진 신경망입니다.
    
    - **1단계:** "I"($x_1$)가 들어오면, 계산을 통해 "기억 1"($h_1$)을 만듭니다.
        
    - **2단계:** "am"($x_2$)이 들어오면, 이 $x_2$와 **"기억 1"($h_1$)**을 함께 사용해 "기억 2"($h_2$)를 만듭니다.
        
    - **3단계:** "a"($x_3$)가 들어오면, 이 $x_3$와 **"기억 2"($h_2$)**를 함께 사용해 "기억 3"($h_3$)을 만듭니다.
        
- 이렇게 매 단계(time step)마다 **이전의 기억($h_{t-1}$)**과 **현재 입력($x_t$)**을 받아 **현재의 기억($h_t$)**을 업데이트합니다.
    
- **개념적 수식:**
    
    - $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
        
        - $h_t$ (현재 기억)는 $h_{t-1}$ (이전 기억)과 $x_t$ (현재 입력)를 조합하여 계산됩니다.
            
        - ($\tanh$는 ReLU나 시그모이드 같은 활성화 함수입니다.)
            
    - $y_t = W_{hy} h_t + b_y$
        
        - (필요하다면) 각 단계의 '기억' $h_t$를 이용해 예측 $y_t$를 만듭니다.
            

**3. PyTorch의 `nn.RNN` 레이어**

- 이 복잡한 루프 계산을 PyTorch는 `nn.RNN`이라는 레이어 하나로 편리하게 제공합니다.
    
- **주요 파라미터:**
    
    - `input_size`: 입력 데이터 1개(시퀀스의 1개 요소)가 몇 개의 숫자로 이루어져 있는지. (예: 원-핫 인코딩된 단어의 차원)
        
    - `hidden_size`: '기억' $h_t$를 몇 개의 숫자로 저장할지 (은닉 상태의 차원, 즉 뉴런 수).
        
    - `batch_first=True`: **(필수 권장!)** 입력 텐서의 순서를 `(배치 크기, 시퀀스 길이, input_size)`로 다룰 수 있게 해주는 편리한 옵션입니다.


RNN의 가장 일반적인 활용 형태 중 하나인 **"Many-to-One"** 문제를 풀어보겠습니다.

- **문제:** 여러 개의 순차적인 데이터(Many)를 보고, 마지막에 **하나의** 결론(One)을 예측합니다.
    
- **예시:** `[8, 1, 5, 3]` 순서 $\rightarrow$ `Class 0 (음수 패턴)`, `[2, 7, 4, 9]` 순서 $\rightarrow$ `Class 1 (양수 패턴)`을 분류하는 (가상의) 문제.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- 0. 데이터 준비 ---
# [Batch, Sequence Length, Input_dim]
# 2개의 데이터 샘플, 각 샘플은 4개의 시퀀스(순서), 각 시퀀스 요소는 1개의 특성
x_train = torch.tensor([
    [[8.0], [1.0], [5.0], [3.0]],  # Sample 1 (Class 0)
    [[2.0], [7.0], [4.0], [9.0]]   # Sample 2 (Class 1)
])
# y_train (각 샘플의 최종 정답)
y_train = torch.tensor([[0.0], [1.0]]) # Sample 1 -> 0, Sample 2 -> 1

# --- 하이퍼파라미터 ---
input_size = 1   # 입력 1개(예: 8.0)의 차원
hidden_size = 5  # '기억(hidden state)'의 크기
num_layers = 1   # RNN 층을 1개만 쌓음
output_size = 1  # 최종 출력 (0 또는 1)

# --- 1. 모델 정의 (class 사용) ---
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # 1. RNN 레이어 정의
        # batch_first=True: 입력 텐서 순서를 (batch_size, seq_length, input_size)로
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # 2. 최종 출력을 위한 Linear 레이어 (Many-to-One)
        # RNN의 마지막 '기억(hidden state)'을 받아서 1개의 출력으로
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 3. 출력을 0~1 사이 확률로
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 초기 은닉 상태(h_0)를 0으로 초기화
        # (layer_dim, batch_size, hidden_dim)
        # x.size(0)은 배치 크기(여기서는 2)
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # 2. RNN 실행
        # rnn_out: 모든 시퀀스 단계(4개)의 hidden state 출력
        # h_n:    마지막 단계(t=4)의 hidden state (기억)
        rnn_out, h_n = self.rnn(x, h_0)
        
        # 3. Many-to-One: 우리는 마지막 '기억' h_n만 필요
        # h_n의 shape: (num_layers, batch_size, hidden_size) -> (1, 2, 5)
        # [0]을 통해 (batch_size, hidden_size)로 변경 -> (2, 5)
        h_n_last = h_n[0] 
        
        # 4. Linear 레이어 통과
        out = self.fc(h_n_last)
        
        # 5. Sigmoid 활성화
        out = self.sigmoid(out)
        
        return out

# 모델 인스턴스(객체) 생성
model = SimpleRNNModel(input_size, hidden_size, num_layers, output_size)

# --- 2. 손실 함수(Loss) 및 옵티마이저(Optimizer) 정의 ---
criterion = nn.BCELoss() # 바이너리 분류
optimizer = optim.SGD(model.parameters(), lr=0.1)

# --- 3. 학습(Training) 실행 ---
epochs = 500

for epoch in range(epochs + 1):
    prediction = model(x_train)
    loss = criterion(prediction, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        predicted_classes = (prediction >= 0.5).float()
        accuracy = (predicted_classes == y_train).float().mean()
        print(f'Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.6f} | Accuracy: {accuracy.item() * 100:.2f}%')

# --- 4. 학습 결과 확인 ---
print("\n--- 학습 완료 후 ---")
final_prediction = (model(x_train) >= 0.5).float()
for i in range(len(x_train)):
    print(f"Input: {x_train[i].view(-1).tolist()} | 정답: {y_train[i].item()} | 예측: {final_prediction[i].item()}")
```