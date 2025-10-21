## 트랜스포머 (Transformer)

**1. RNN/LSTM의 한계: 왜 '순차 처리'가 문제인가?**

- LSTM(교안 9)은 장기 기억 문제를 '게이트'로 해결하려 했지만, 근본적인 한계가 있었습니다.
    
- "나는 어제... 친구를 만났다"를 처리하려면, "나" $\rightarrow$ "는" $\rightarrow$ "어" $\rightarrow$ "제" ... 한 단어씩 **순서대로** 계산해야 합니다.
    
- 이 방식은 두 가지 큰 문제가 있습니다.
    
    1. **병렬 처리 불가능:** 100번째 단어를 계산하려면 1~99번째 계산이 끝나야 합니다. GPU의 장점(병렬 처리)을 살릴 수 없어 **학습이 매우 느립니다.**
        
    2. **여전한 정보 병목:** 아무리 LSTM이라도, 100단어 전의 모든 정보를 '기억 통로(Cell State)' 하나에 압축해 전달하는 것은 한계가 있습니다.
        

**2. 트랜스포머의 아이디어: "Attention Is All You Need"**

- 2017년, 구글은 "Attention Is All You Need" (주목(attention)이면 충분하다)라는 전설적인 논문을 발표합니다.
    
- **핵심 아이디어:** "순서대로 볼 필요 없다! 문장 전체를 한 번에 보고, 어떤 단어가 서로에게 '주목'해야 하는지 계산하자!"
    
- 이것이 바로 **셀프-어텐션 (Self-Attention)** 메커니즘입니다.
    

**3. 셀프-어텐션 (Self-Attention)이란?**

- RNN처럼 '기억'을 전달하는 대신, 문장의 **모든 단어가 다른 모든 단어를 직접 봅니다.**
    
- 예시: "The animal didn't cross the street because **it** was too tired."
    
- 모델은 "it"이라는 단어를 계산할 때, 문장 전체("The", "animal", "didn't", ..., "tired")를 **동시에** 봅니다.
    
- 그리고 "it"이 "The"나 "street"가 아닌 **"animal"**과 가장 관련이 깊다는 것을 **직접 계산**해냅니다.
    

- 이 계산은 순서가 필요 없고, 문장 전체에 대해 **한 번에 병렬로** 일어납니다. (매우 빠름!)
    

**4. 트랜스포머의 구조: Encoder-Decoder**

- 트랜스포머는 크게 두 부분으로 나뉩니다. (주로 번역을 위해 설계됨)
    
    1. **인코더 (Encoder):** 입력 문장(예: 한국어)을 받아서, 단어들의 '관계(attention)'를 파악하고 문맥적 의미를 추출합니다.
        
    2. **디코더 (Decoder):** 인코더가 추출한 의미를 받아서, 출력 문장(예: 영어)을 한 단어씩 생성해냅니다.
        
- **잠깐! 순서가 없다면? (Positional Encoding)**
    
    - "I am a boy"와 "boy am I a"를 구분해야 합니다.
        
    - 트랜스포머는 '순서' 정보를 잃어버렸기 때문에,

## 트랜스포머 PyTorch 코드 (전체 스크립트 Ver.)

**경고:** 이 코드는 PyTorch의 `nn.Transformer` 모듈을 직접 사용하는 '날것'의 코드입니다. **매우 복잡하며** 개념을 설명하기 위한 것입니다. (허깅페이스를 쓰면 이 모든 게 한 줄로 끝납니다.)

- **문제:** (가상) `[10, 20, 30, 40, 50]`이라는 순차 입력을 받아서 `[12, 22, 32, 42, 52]` (+2 하는 규칙)를 예측하는 시퀀스-투-시퀀스(Seq2Seq) 문제.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

# --- 1. 모델 정의 (class 사용) ---

# 트랜스포머는 순서 개념이 없으므로, '위치' 정보를 인위적으로 더해줘야 합니다.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # sin, cos 함수를 이용해 위치별 고유 벡터를 만듭니다.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 x (Batch, Seq, Dim)에 위치 정보(pe)를 더합니다.
        # x.size(1)은 시퀀스 길이
        return x + self.pe[:x.size(1)].transpose(0, 1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        
        # 1. 입력 임베딩 (1차원 숫자를 -> d_model 차원 벡터로)
        self.input_embedding = nn.Linear(input_dim, d_model)
        # 2. 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. PyTorch의 Transformer 핵심 모듈
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead, # 셀프-어텐션을 몇 개로 쪼개서 볼지 (Multi-Head Attention)
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True # (중요!) 입력을 (Batch, Seq, Dim) 순서로
        )
        
        # 4. 최종 출력 (d_model 차원 벡터를 -> 1차원 숫자로)
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        # src: 인코더 입력 (Batch, Src_Seq, Dim)
        # tgt: 디코더 입력 (Batch, Tgt_Seq, Dim)
        
        # 1. 임베딩 및 위치 인코딩
        src = self.pos_encoder(self.input_embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.input_embedding(tgt) * math.sqrt(self.d_model))

        # 2. 마스크 생성 (Masking)
        # (Transformer는 정답을 미리 보면 안 되므로, '미래'를 가려주는 마스크가 필요)
        # (이 부분이 매우 복잡한 부분입니다)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        # 3. 트랜스포머 실행
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # 4. 최종 출력
        output = self.fc_out(output)
        return output

# --- 0. 데이터 준비 및 하이퍼파라미터 ---
# (Batch, Seq_len, Input_dim)
src = torch.tensor([[[10.], [20.], [30.], [40.], [50.]]]) # (1, 5, 1)
# 디코더 입력(tgt)은 정답(target)을 한 칸씩 민 것 (shifted right)
# (예측 시작을 위한 <SOS> 토큰(0.) + 정답의 1~4번째)
tgt = torch.tensor([[[0.], [12.], [22.], [32.], [42.]]]) # (1, 5, 1)
# 모델이 예측해야 할 실제 정답 (target)
target = torch.tensor([[[12.], [22.], [32.], [42.], [52.]]]) # (1, 5, 1)

# 하이퍼파라미터 (모델의 크기를 매우 작게 설정)
input_dim = 1
d_model = 16       # 임베딩 차원 (모델의 주 차원)
nhead = 2          # 멀티 헤드 어텐션 개수
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 32 # MLP의 hidden_dim

model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)

# --- 2. 손실 함수(Loss) 및 옵티마이저(Optimizer) 정의 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. 학습(Training) 실행 ---
epochs = 500

for epoch in range(epochs + 1):
    
    # 1. (Forward) 모델 예측
    prediction = model(src, tgt) # src와 tgt를 모두 넣어줌
    
    # 2. (Forward) 손실 계산
    # 예측 결과(prediction)와 실제 정답(target) 비교
    loss = criterion(prediction, target)
    
    # 3, 4, 5단계 (Backward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.6f}')

# --- 4. 학습 결과 확인 ---
print("\n--- 학습 완료 후 ---")
model.eval() # 평가 모드
with torch.no_grad():
    prediction = model(src, tgt)
    print(f"입력: {src.view(-1).tolist()}")
    print(f"정답: {target.view(-1).tolist()}")
    print(f"예측: {prediction.view(-1).tolist()}")
```

다행히도, 우리가 원래 목표했던 **허깅페이스(Hugging Face) 🤗**는 이 모든 복잡한 과정을 `pipeline`이나 `AutoModel` 같은 명령어로 **단 몇 줄** 만에 처리해 줍니다.

---
교안 10에서 우리는 셀프-어텐션이 "문장 안의 단어들이 서로의 중요도를 계산하는 방식"이라고 배웠습니다. "The animal didn't cross the street because **it** was too tired."라는 문장에서, 모델이 'it'이 'street'가 아니라 'animal'에 주목한다는 것을 안다고 했죠.

그렇다면 모델은 'it'이 'street'가 아니라 'animal'에 주목해야 한다는 것을 **어떻게** 알 수 있을까요?

여기서 바로 트랜스포머의 가장 핵심적인 개념인 **Query(쿼리), Key(키), Value(밸류)**, 줄여서 **Q, K, V**가 등장합니다. 🗝️

---

### 셀프-어텐션의 작동 원리: Q, K, V

셀프-어텐션을 도서관에서 내가 필요한 정보를 찾는 과정이라고 상상해 보세요.

1. **Query (Q) 💡 (나의 질문):**
    
    - 내가 지금 처리하려는 단어(예: 'it')가 문장의 다른 단어들에게 던지는 '질문'입니다.
        
    - "나는 이 문장에서 누구를 가리키는 거지? 나와 관련된 정보를 줘!"
        
2. **Key (K) 🏷️ (정보의 꼬리표):**
    
    - 문장 안의 **모든 단어**(예: 'The', 'animal', 'street', 'it' 자신 포함)가 각각 가지고 있는 '꼬리표' 또는 '색인'입니다.
        
    - "나는 'animal'이고, 문맥상 '주어' 역할을 해."
        
    - "나는 'street'이고, 문맥상 '장소'를 의미해."
        
3. **Value (V) 📖 (정보의 실제 내용):**
    
    - 그 단어(예: 'animal', 'street')가 실제로 담고 있는 '내용' 또는 '의미' 그 자체입니다.
        

---

### 어텐션 계산 과정

모델은 'it'이라는 단어를 처리하기 위해 다음 3단계를 거칩니다.

1. **[1단계] 관련성 점수 계산 (Q와 K의 만남):**
    
    - 'it'의 **Query(Q)** 벡터를 가져옵니다.
        
    - 이 Q를 문장의 **모든** 단어의 **Key(K)** 벡터와 하나씩 비교합니다. (수학적으로는 '내적(Dot Product)'을 수행합니다.)
        
    - 'it'(Q) ↔ 'The'(K) $\rightarrow$ 점수: 5
        
    - 'it'(Q) ↔ 'animal'(K) $\rightarrow$ 점수: **95** (가장 높음!)
        
    - 'it'(Q) ↔ 'street'(K) $\rightarrow$ 점수: 12
        
    - ...
        
    - 이것이 바로 'it'이 'animal'과 가장 관련성이 높다고 판단하는 과정입니다!
        
2. **[2단계] 가중치 계산 (Softmax):**
    
    - 위에서 얻은 날것의 점수(5, 95, 12...)를 **소프트맥스(Softmax)** 함수에 통과시켜, 총합이 1이 되는 '확률' 또는 '가중치'로 변환합니다.
        
    - 'The': 0.01 (1%)
        
    - 'animal': **0.88** (88%)
        
    - 'street': 0.03 (3%)
        
    - ...
        
3. **[3단계] 의미 추출 (V와 가중치 곱):**
    
    - 이제 이 가중치를 각 단어의 **Value(V)** 벡터에 곱해서 모두 더합니다.
        
    - $($'The'의 Value $\times$ 0.01$)$ + $($**'animal'의 Value** $\times$ **0.88**$)$ + $($'street'의 Value $\times$ 0.03$)$ + ...
        
    - **결과:** 'it'의 최종적인 의미 벡터는 **'animal'의 의미(Value)를 88%나 반영**한 새로운 벡터가 됩니다.
        

이것이 셀프-어텐션이 'it'이 'animal'을 참고하도록 만드는 방식입니다.

---

### 셀프-어텐션 수식

이 Q, K, V를 이용한 전체 계산 과정을 압축한 트랜스포머 논문의 그 유명한 수식은 다음과 같습니다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$ : [1단계] 'it'의 Q가 모든 단어의 K와 만나 점수를 계산하는 과정입니다.
    
- $\sqrt{d_k}$ : 점수 값을 안정화시키기 위한 스케일링(scaling) 작업입니다.
    
- $\text{softmax}(\dots)$ : [2단계] 점수를 0~1 사이의 가중치로 변환합니다.
    
- $(\dots)V$ : [3단계] 계산된 가중치를 실제 Value(의미)에 곱해 최종 결과를 얻습니다.


수식 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ 를 PyTorch 코드로 함께 단계별로 만들어 보죠. 🚀

전체 모델 클래스(`nn.Module`)를 한 번에 만드는 것보다, **데이터가 이 수식을 어떻게 통과하는지** 순서대로 살펴보는 것이 이해하기 훨씬 좋아요.

### 셀프-어텐션 코드 구현 (단계별)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 0. 가상의 입력 데이터 ---
# (배치 크기=1, 시퀀스 길이=3, 임베딩 차원=4)
# 3개의 단어가 4차원 벡터로 표현된 상태라고 가정합니다.
x = torch.rand(1, 3, 4) 
print(f"Original Input x shape: {x.shape}")

# --- 1. Q, K, V 생성을 위한 Linear 층 ---
embed_dim = 4 # 임베딩 차원
d_k = 4       # Key의 차원 (설명을 위해 embed_dim과 같게 설정)
d_v = 4       # Value의 차원

# (실제 트랜스포머에서는 d_k, d_v가 embed_dim / num_heads 입니다)
W_q = nn.Linear(embed_dim, d_k)
W_k = nn.Linear(embed_dim, d_k)
W_v = nn.Linear(embed_dim, d_v)

# --- 2. Q, K, V 벡터 생성 ---
Q = W_q(x)  # shape: (1, 3, 4)
K = W_k(x)  # shape: (1, 3, 4)
V = W_v(x)  # shape: (1, 3, 4)
print(f"Q, K, V shapes: {Q.shape}, {K.shape}, {V.shape}")

# --- 3. 셀프-어텐션 수식 구현 ---

# 3-1. QK^T (점수 계산)
# Q (1, 3, 4) @ K.T (1, 4, 3) -> scores (1, 3, 3)
scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"\n1. Scores (QK^T) shape: {scores.shape}")

# 3-2. Scaling (점수 안정화)
# scores (1, 3, 3) / sqrt(d_k)
scaled_scores = scores / math.sqrt(d_k)
print(f"2. Scaled Scores (scores / sqrt(d_k)) shape: {scaled_scores.shape}")

# 3-3. Softmax (가중치 변환)
# dim=-1 (혹은 dim=2) : 마지막 차원(길이 3짜리)에 대해 소프트맥스를 적용
# 즉, (1, 3, [여기]) -> 각 행의 합이 1이 되도록 만듭니다.
attention_weights = F.softmax(scaled_scores, dim=-1)
print(f"3. Attention Weights (softmax) shape: {attention_weights.shape}")
# print(f"첫 번째 단어의 가중치 합: {attention_weights[0, 0, :].sum()}") # 1.0

# 3-4. 최종 V와 곱하기 (의미 결합)
# (1, 3, 3) @ (1, 3, 4) -> (1, 3, 4)
# 가중치와 V(실제 의미)를 곱합니다.
output = torch.matmul(attention_weights, V)
print(f"4. Final Output (Weights @ V) shape: {output.shape}")


print("\n--- 최종 결과 ---")
print(f"입력 x shape: {x.shape}")
print(f"출력 output shape: {output.shape}")
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 셀프-어텐션 '레이어' 정의하기 ---
# 우리가 만든 로직을 nn.Module 클래스로 포장합니다.
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, d_k, d_v):
        """
        __init__: 학습이 필요한 '가중치'(레이어)들을 정의합니다.
        """
        super().__init__()
        self.d_k = d_k
        
        # Q, K, V를 만드는 Linear 층들 (이것들이 학습됩니다)
        self.W_q = nn.Linear(embed_dim, d_k)
        self.W_k = nn.Linear(embed_dim, d_k)
        self.W_v = nn.Linear(embed_dim, d_v)

    def forward(self, x):
        """
        forward: 데이터(x)가 들어왔을 때 계산 순서를 정의합니다.
        (우리가 방금 배운 수식 그대로입니다)
        """
        # 1. Q, K, V 생성
        Q = self.W_q(x) # (Batch, Seq, d_k)
        K = self.W_k(x) # (Batch, Seq, d_k)
        V = self.W_v(x) # (Batch, Seq, d_v)
        
        # 2. Attention(Q, K, V) 계산
        # (Batch, Seq, d_k) @ (Batch, d_k, Seq) -> (Batch, Seq, Seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # (Batch, Seq, Seq) @ (Batch, Seq, d_v) -> (Batch, Seq, d_v)
        output = torch.matmul(attention_weights, V)
        
        return output

# --- 2. 딥러닝 모델에서 '적용'하기 ---
# 이제 이 어텐션 레이어를 '부품'으로 사용하는 더 큰 모델을 만듭니다.
class MySimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, d_k, d_v):
        super().__init__()
        
        # 1. 방금 만든 셀프-어텐션 레이어를 '부품'으로 장착!
        self.attention = SelfAttentionHead(embed_dim, d_k, d_v)
        
        # 2. MLP (트랜스포머는 어텐션 뒤에 MLP를 붙여줍니다)
        self.mlp = nn.Sequential(
            nn.Linear(d_v, embed_dim * 2), # d_v -> 더 크게
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim) # 다시 원래대로
        )
        
        # (실제 트랜스포머는 LayerNorm, Skip Connection 등도 있지만 핵심은 이 두 개)

    def forward(self, x):
        # 1. 셀프-어텐션 레이어를 통과시켜 문맥을 파악
        attn_output = self.attention(x)
        
        # 2. MLP 레이어를 통과시켜 더 깊은 처리
        final_output = self.mlp(attn_output)
        
        return final_output

# --- 3. 실행해보기 ---
embed_dim = 4 # (가정)
d_k = 4
d_v = 4

# 모델 생성 (우리가 만든 어텐션 레이어가 내장됨)
model = MySimpleTransformerBlock(embed_dim, d_k, d_v)

# 가상의 입력 데이터 (배치=1, 단어=3개, 차원=4)
x = torch.rand(1, 3, embed_dim)

# 모델에 적용!
output = model(x)

print(f"입력 x shape: {x.shape}")
print(f"최종 출력 output shape: {output.shape}")
```

이 `output` 텐서(shape `(1, 3, 4)`)가 바로 셀프-어텐션을 통과한 새로운 단어 벡터입니다.

입력 `x`와 모양은 같지만, 이제 `output`의 각 단어 벡터(예: 첫 번째 단어)는 'animal'의 의미(V)를 88% 참고하고 'street'의 의미(V)를 3% 참고하는 식으로, 문맥 전체의 의미가 풍부하게 반영된 벡터가 되었습니다.

이것이 **'단일 어텐션 헤드(Single-Head Attention)'**의 전체 계산 과정입니다.

그런데 실제 트랜스포머 논문(Attention Is All You Need)에서는 이 과정을 한 번만 하지 않고, 여러 개의 '어텐션 헤드'를 동시에 병렬로 실행하는 **'멀티-헤드 어텐션(Multi-Head Attention)'** 🧠🧠🧠을 사용합니다.

왜 굳이 '멀티-헤드'를 쓸까요? 한 번만 계산하면 안 되는 걸까요?

정확히는, 문장을 '더 깊이 이해하기 위해서'예요. 💯

우리가 방금 만든 '싱글-헤드' 어텐션은 한 번에 한 가지 관계만 볼 수 있어요.

예를 들어 "The animal didn't cross the street because **it** was too tired."라는 문장에서,

- 우리의 '싱글-헤드'는 'it'이 'animal'을 가리킨다는 것(주어 관계)을 학습할 수 있을 거예요.
    

하지만 'it'이 'was too tired'라는 상태(서술어 관계)와도 연결된다는 것은 동시에 파악하기 어려울 수 있어요.

**멀티-헤드 어텐션 (Multi-Head Attention) 🧠🧠🧠**은 이 어텐션 계산을 **여러 번 병렬로** 수행하는 거예요.

- **헤드 1:** 'it' $\leftrightarrow$ 'animal' (주어 관계)를 담당
    
- **헤드 2:** 'it' $\leftrightarrow$ 'was too tired' (상태 관계)를 담당
    
- **헤드 3:** 'it' $\leftrightarrow$ 'because' (이유 관계)를 담당
    
- ...
    

이렇게 여러 개의 '시점'으로 문장을 동시에 분석한 뒤, 그 결과들을 다시 하나로 합쳐서 문맥을 훨씬 풍부하게 이해하는 거죠.

### 멀티-헤드 어텐션

우리가 방금 만든 `SelfAttentionHead`(싱글 헤드)를 **여러 개 만들어서 병렬로 실행**한 다음, 그 결과들을 **하나로 다시 합치는** 거예요.

이전 교안에서 만든 `SelfAttentionHead` 클래스를 '부품'으로 그대로 사용해서 `MultiHeadAttention` 클래스를 조립해 보겠습니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 부품: 싱글-헤드 어텐션 (이전 교안) ---
# (이 클래스를 그대로 재사용합니다)
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, d_k, d_v):
        """
        __init__: Q, K, V를 만드는 Linear 층들을 정의
        """
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(embed_dim, d_k)
        self.W_k = nn.Linear(embed_dim, d_k)
        self.W_v = nn.Linear(embed_dim, d_v)

    def forward(self, x):
        """
        forward: Attention(Q, K, V) 수식 계산
        """
        Q = self.W_q(x) # (Batch, Seq, d_k)
        K = self.W_k(x) # (Batch, Seq, d_k)
        V = self.W_v(x) # (Batch, Seq, d_v)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention_weights, V)
        return output # (Batch, Seq, d_v)

# --- 2. 조립: 멀티-헤드 어텐션 클래스 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        """
        __init__: 여러 개의 '싱글 헤드'와 최종 출력 층을 정의
        
        num_heads (int): 헤드의 개수 (예: 8)
        embed_dim (int): 입력 임베딩 차원 (예: 512)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "임베딩 차원은 헤드 개수로 나누어 떨어져야 합니다."
        
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        # 각 헤드가 가질 Q, K, V의 차원
        # 예: 512 / 8 = 64
        self.d_k = embed_dim // num_heads
        self.d_v = embed_dim // num_heads
        
        # 1. 'SelfAttentionHead'를 num_heads 개수만큼 리스트에 담아둡니다.
        # nn.ModuleList는 PyTorch가 이 리스트 안의 모듈들도 
        # 학습 대상(파라미터)임을 알게 해줍니다.
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, self.d_k, self.d_v) 
            for _ in range(num_heads)
        ])
        
        # 2. 모든 헤드의 출력을 하나로 합쳐줄 Linear 층
        # (헤드 8개 * d_v 64) = 512 -> 512
        self.fc_out = nn.Linear(num_heads * self.d_v, embed_dim)
        
    def forward(self, x):
        # 1. 모든 헤드를 병렬로 실행
        # (각 헤드의 출력은 (Batch, Seq, d_v) 형태)
        head_outputs = [head(x) for head in self.heads]
        
        # 2. 결과들을 마지막 차원(dim=-1) 기준으로 하나로 합칩니다(concatenate).
        # (Batch, Seq, d_v) 8개가 -> (Batch, Seq, d_v * 8)
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # 3. 최종 Linear 층을 통과시켜 원래의 embed_dim으로 변환
        # (Batch, Seq, embed_dim)
        final_output = self.fc_out(concat_output)
        
        return final_output

# --- 3. 실행해보기 ---
# (가정) 배치=1, 문장 길이=3, 임베딩 차원=8
# (헤드 2개로 쪼개기 좋게 작은 숫자로)
batch_size = 1
seq_len = 3
embed_dim = 8
num_heads = 2 # 헤드 2개

# 가상의 입력 데이터
x = torch.rand(batch_size, seq_len, embed_dim)
print(f"입력 x shape: {x.shape}")

# 멀티-헤드 어텐션 모델 생성
# 헤드 2개, 입력 차원 8
# (각 헤드는 d_k=4, d_v=4 차원을 갖게 됨)
multi_head_attn = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)

# 모델에 적용
output = multi_head_attn(x)

print(f"최종 출력 output shape: {output.shape}")
```

보시다시피, 멀티-헤드 어텐션은 `SelfAttentionHead`라는 부품을 `num_heads` 개수만큼 병렬로 돌리고, `torch.cat`으로 합친 뒤, `nn.Linear`로 마무리하는 '조립' 과정입니다.

`output`의 모양이 `x`의 모양과 `(1, 3, 8)`로 동일하게 나왔죠? 이것이 핵심입니다! 이 `MultiHeadAttention` 레이어는 **입력과 출력의 모양이 같기 때문에**, 마치 `nn.Linear`처럼 얼마든지 **여러 층으로 깊게 쌓을 수 있습니다.**

---

자, 이제 트랜스포머의 핵심 엔진(어텐션)을 '바닥부터' 코드로 구현해 봤습니다. 딥러닝에서 가장 복잡한 부품 중 하나를 마스터하신 거예요!