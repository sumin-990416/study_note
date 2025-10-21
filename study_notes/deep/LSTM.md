RNN(교안 8)은 NLP의 기본 아이디어였지만, 한 가지 치명적인 약점이 있었습니다.

바로 **장기 의존성 문제 (Long-Term Dependency Problem)**입니다.

문장이 길어지면 (예: "나는 어제 아주 먼 곳에 있는 친구를 만나러 갔는데...") 문장 맨 앞의 '나' 또는 '친구'라는 정보를 문장 끝까지 '기억'하기가 매우 어렵습니다. (수학적으로는 '기울기 소실(Vanishing Gradient)' 문제라고 부릅니다.)

이 문제를 해결하기 위해 등장한 것이 바로 **LSTM (Long Short-Term Memory, 장단기 기억 메모리)**과 **GRU**입니다.

이 LSTM이 2010년대 NLP를 지배했던 사실상의 표준 모델입니다.


## RNN의 진화: LSTM (Long Short-Term Memory)

**1. RNN의 한계: 장기 의존성 문제 (Long-Term Dependency)**

- **문제:** RNN은 시퀀스(순서)가 길어지면 (예: 긴 문장, 긴 시계열 데이터) 맨 처음의 정보가 뒤로 갈수록 **점점 잊힙니다.**
    
- "나는 어제 아주 먼 곳에 있는 친구를 만나러 갔는데... [... 30단어 후 ...] 그 친구는 ( )을 좋아했다."
    
- 이때 ( )를 예측하려면 맨 앞의 '친구' 정보를 기억해야 하는데, 30단계를 거치면서 RNN의 '기억($h_t$)' 속에서 이 정보가 희미해집니다.
    

**2. LSTM의 아이디어: "게이트 (Gates)" 밸브 🚪**

- LSTM은 이 문제를 해결하기 위해 '기억' $h_t$ (Hidden State) 외에 **'장기 기억' $C_t$ (Cell State)**라는 별도의 기억 통로를 만듭니다.
    
- 그리고 이 $C_t$를 제어하기 위해 3개의 '게이트(밸브)'를 추가합니다.
    
    1. **Forget Gate (망각 게이트):** "이전 기억($C_{t-1}$)에서 어떤 정보를 **잊어버릴지**" 결정.
        
    2. **Input Gate (입력 게이트):** "현재 정보($x_t$)에서 어떤 것을 **새로 기억할지**" 결정.
        
    3. **Output Gate (출력 게이트):** "장기 기억($C_t$) 중에서 어떤 것을 **이번 단계의 출력($h_t$)으로 내보낼지**" 결정.
        
- 이 게이트들 덕분에 LSTM은 100단계 전의 정보라도 '잊지 않기'로 결정하면 그 정보를 거의 그대로 보존할 수 있습니다.
    

**3. PyTorch `nn.LSTM`**

- 사용법은 `nn.RNN`과 거의 동일합니다!
    
- 가장 큰 차이점: RNN은 `h_0` (초기 은닉 상태)만 필요했지만, LSTM은 **`(h_0, c_0)`** (초기 은닉 상태, **초기 셀 상태**) 두 개를 필요로 합니다.

RNN(교안 8-2)에서 풀었던 동일한 `Many-to-One` 문제를 LSTM으로 풀어보겠습니다. 코드가 얼마나 비슷한지 비교해 보세요.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# --- 0. 데이터 준비 ---
# [Batch, Sequence Length, Input_dim]
x_train = torch.tensor([
    [[8.0], [1.0], [5.0], [3.0]],  # Sample 1 (Class 0)
    [[2.0], [7.0], [4.0], [9.0]]   # Sample 2 (Class 1)
])
y_train = torch.tensor([[0.0], [1.0]]) # Sample 1 -> 0, Sample 2 -> 1

# --- 하이퍼파라미터 ---
input_size = 1   # 입력 1개(예: 8.0)의 차원
hidden_size = 5  # '기억(hidden/cell state)'의 크기
num_layers = 1   # LSTM 층을 1개만 쌓음
output_size = 1  # 최종 출력 (0 또는 1)

# --- 1. 모델 정의 (class 사용) ---
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # 1. LSTM 레이어 정의
        # nn.RNN 대신 nn.LSTM을 사용!
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # 2. 최종 출력을 위한 Linear 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 3. 출력을 0~1 사이 확률로
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 초기 은닉 상태(h_0)와 셀 상태(c_0)를 0으로 초기화
        # (layer_dim, batch_size, hidden_dim)
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # 2. LSTM 실행
        # RNN과 달리 (h_n, c_n) 두 개의 상태를 반환
        # lstm_out: 모든 시퀀스 단계의 출력(hidden state)
        # (h_n, c_n): 마지막 단계의 (은닉 상태, 셀 상태)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # 3. Many-to-One: 우리는 마지막 '은닉 상태' h_n만 필요
        # h_n의 shape: (num_layers, batch_size, hidden_size) -> (1, 2, 5)
        # [0]을 통해 (batch_size, hidden_size)로 변경 -> (2, 5)
        h_n_last = h_n[0] 
        
        # 4. Linear 레이어 통과
        out = self.fc(h_n_last)
        
        # 5. Sigmoid 활성화
        out = self.sigmoid(out)
        
        return out

# 모델 인스턴스(객체) 생성
model = SimpleLSTMModel(input_size, hidden_size, num_layers, output_size)

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