## 허깅페이스 (Hugging Face)

**1. 허깅페이스(Hugging Face)란?**

- 트랜스포머(Transformer) 모델(예: BERT, GPT)을 누구나 쉽게 다운로드하고, 학습시키고, 사용할 수 있도록 만든 **AI 모델의 거대한 허브(저장소)이자 라이브러리**입니다.
    
- PyTorch로 `nn.Transformer`를 직접 만드는 것이 '자동차 부품으로 차를 조립'하는 것이라면, 허깅페이스는 '완성된 슈퍼카의 키를 받아 바로 운전'하는 것과 같습니다.
    

**2. 핵심 기능 1: `pipeline` (가장 쉬운 방법)**

- 허깅페이스를 시작하는 가장 쉽고 빠른 방법입니다.
    
- `pipeline`은 **"어떤 작업(task)"**을 하고 싶은지만 말해주면, 알아서 그 작업에 가장 적합한 모델과 데이터 처리 방식(토크나이저)을 **자동으로** 다운로드하고 실행해 줍니다.
    
- 우리가 교안 7에서 `DataLoader`로 파이프라인을 짰던 것처럼, `pipeline`은 **"데이터 입력 $\rightarrow$ 모델 예측 $\rightarrow$ 결과 출력"**의 전 과정을 하나로 합쳐줍니다.
    
- **주요 `pipeline` 작업(task) 예시:**
    
    - `sentiment-analysis`: 감성 분석 ("이 문장은 긍정인가 부정인가?")
        
    - `text-generation`: 텍스트 생성 (예: GPT)
        
    - `fill-mask`: 빈칸 채우기 (예: "나는 [MASK]에 간다.") (BERT의 주특기)
        
    - `translation_xx_to_yy`: 번역

```python
# pip install transformers
import torch
# 🌟 허깅페이스 transformers 라이브러리에서 pipeline을 가져옵니다.
from transformers import pipeline

# --- 1. 파이프라인 생성 ---
# "감성 분석(sentiment-analysis)" 작업을 수행하는 파이프라인을 로드합니다.
# (이때, 허깅페이스가 이 작업에 적합한 기본 모델을 알아서 다운로드합니다.)
print("파이프라인 로드 중...")
sentiment_pipeline = pipeline(task="sentiment-analysis")
print("로드 완료!")

# --- 2. 데이터 준비 ---
my_text1 = "This movie is so great and wonderful!"
my_text2 = "I hate this restaurant. The food was terrible."
my_texts = [my_text1, my_text2]

# --- 3. 파이프라인 실행 (모델 예측) ---
# 복잡한 DataLoader나 model(x)가 필요 없습니다.
# 그냥 파이프라인에 데이터를 넣으면 끝입니다.
results = sentiment_pipeline(my_texts)

# --- 4. 결과 확인 ---
print("\n--- 예측 결과 ---")
for text, result in zip(my_texts, results):
    print(f"입력: {text}")
    print(f"결과: Label={result['label']}, Score={result['score']:.4f}")

# --- 예제 2: 빈칸 채우기 (Fill-Mask) ---
print("\n--- 빈칸 채우기 파이프라인 ---")
fill_mask_pipeline = pipeline(task="fill-mask")
text_with_mask = "The capital of France is [MASK]."
mask_results = fill_mask_pipeline(text_with_mask)

print(f"입력: {text_with_mask}")
for result in mask_results:
    # (score가 가장 높은 상위 5개 정도를 보여줍니다)
    print(f"예측: {result['token_str']} (Score: {result['score']:.4f})")
```

`pipeline`은 정말 편리하지만, '블랙 박스'와 같아서 우리가 가진 새로운 데이터로 모델을 _학습_시킬 수는 없어요.

그렇게 하려면, `pipeline`이 자동으로 해줬던 두 가지 핵심 단계를 우리가 직접 수행해야 해요.

1. **토크나이저 (Tokenizer) 📖:** 모델(BERT, GPT 등)은 "안녕하세요" 같은 텍스트를 이해하지 못해요. 오직 숫자만 이해할 수 있죠. **토크나이저**는 우리의 텍스트를, 그 모델이 사전에 학습했던 방식과 **똑같은** 숫자(토큰) 형식으로 변환해 주는 '번역기'입니다.
    
2. **모델 (Model) 🧠:** 교안 10에서 배운, 실제 사전 학습된 트랜스포머 모델('뇌')입니다.
    

허깅페이스의 마법은 바로 `Auto` 클래스에 있습니다. 우리는 그저 허깅페이스 허브에 있는 모델의 이름(ID, 예: `"bert-base-uncased"`)만 알려주면, `AutoTokenizer`와 `AutoModel`이 알아서 그 이름에 맞는 '토크나이저'와 '모델' 한 쌍을 다운로드해 줍니다.

---

## `AutoTokenizer` & `AutoModel` 

- 이 코드는 사전 학습된 모델(BERT)을 수동으로 로드하고 사용하여 원시 출력(raw output)을 얻는 방법을 보여줍니다.
    
- (실행 전 `pip install transformers` 필요)

```python
import torch
# 🌟 AutoTokenizer와 AutoModel을 가져옵니다.
from transformers import AutoTokenizer, AutoModel

# --- 1. 모델 이름(ID) 지정 ---
# 허깅페이스 Hub에서 사용할 모델의 이름을 정합니다.
# "bert-base-uncased"는 가장 기본이 되는 (영어) BERT 모델입니다.
# (만약 한글을 다루고 싶다면, "klue/bert-base" 같은 한글 모델 이름을 사용합니다)
MODEL_NAME = "bert-base-uncased" 

# --- 2. Tokenizer 로드 ---
# MODEL_NAME에 해당하는 '단어장'과 '토큰화 규칙'을 다운로드합니다.
print(f"'{MODEL_NAME}'의 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 3. Model 로드 ---
# MODEL_NAME에 해당하는 '모델 구조'와 '사전 학습된 가중치(파라미터)'를 다운로드합니다.
print(f"'{MODEL_NAME}'의 모델 로드 중...")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval() # 학습이 아닌, 추론(inference) 모드로 설정

# --- 4. 데이터 준비 및 토큰화 ---
my_texts = [
    "Hugging Face is a great library.",
    "Transformers are powerful."
]

print(f"\n원본 텍스트: {my_texts}")

# 토크나이저를 실행합니다.
# padding=True: 문장 길이를 맞추기 위해 짧은 문장에 [PAD] 토큰 추가
# truncation=True: 모델이 처리할 수 있는 최대 길이를 넘으면 자르기
# return_tensors='pt': 결과를 PyTorch 텐서로 반환
inputs = tokenizer(my_texts, padding=True, truncation=True, return_tensors="pt")

# 'inputs'는 딕셔너리(dictionary) 형태입니다.
# print(f"토큰화 결과 (inputs): \n{inputs}")
# 'input_ids'는 텍스트가 변환된 숫자 텐서입니다.
print(f"토큰 ID (input_ids): \n{inputs['input_ids']}")

# --- 5. 모델 예측 (Forward Pass) ---
# 모델에 토큰화된 텐서를 넣습니다.
# torch.no_grad()는 "미분(gradient) 계산을 하지 않겠다" (추론 시 메모리 절약)
with torch.no_grad():
    # **inputs는 딕셔너리의 모든 키-값을 (input_ids=..., attention_mask=...)로
    # 풀어서 모델에 전달하는 Python 문법입니다.
    outputs = model(**inputs)

# --- 6. 결과 확인 ---
# 'outputs.last_hidden_state'가 모델의 최종 출력입니다.
# 이것이 바로 텍스트의 '문맥적 의미'가 압축된 고차원 벡터입니다!
last_hidden_state = outputs.last_hidden_state

print(f"\n모델 최종 출력 (last_hidden_state)의 Shape:")
# (Batch Size, Sequence Length, Hidden Dim)
# (데이터 2개, 토큰 8개, BERT-base의 은닉 차원 768)
print(last_hidden_state.shape)
```


## 미세조정 (Fine-Tuning)

**1. 미세조정(Fine-Tuning)이란?**

- **개념:** 이미 수백만, 수십억 개의 텍스트로 '세상의 일반적인 지식'을 학습한 **사전 학습된(pre-trained)** 트랜스포머 모델(예: BERT, GPT)을 가져옵니다.
    
- 그리고 이 거대한 모델을 **우리의 특정 작업**(예: 영화 리뷰 긍/부정 분류, 스팸 메일 분류)에 맞게 **'살짝' 추가로 학습**시키는 과정입니다.
    
- 비유: 영어 원어민(사전 학습된 모델)에게 '의학 용어'(우리 데이터)를 가르쳐서 '의학 전문 번역가'(미세조정된 모델)로 만드는 것과 같습니다.
    

**2. `AutoModel` vs. `AutoModelForSequenceClassification`**

- **교안 11:** `AutoModel`을 썼습니다.
    
    - 이건 트랜스포머의 '몸통'만 가져온 거예요.
        
    - 출력은 `last_hidden_state`(문맥이해 벡터)였습니다.
        
    - 이걸로 분류를 하려면, 우리가 직접 `nn.Linear`를 뒤에 붙여야 했죠.
        
- **교안 13 (지금):** `AutoModelForSequenceClassification`을 씁니다.
    
    - 이것은 '몸통' + '분류 작업용 머리(`nn.Linear`)'가 **이미 결합된** 완성품 모델입니다.
        
    - 우리는 그저 이 모델에게 "정답은 2개야(예: 긍정/부정)"라고 알려주기만 하면 됩니다.
        

**3. 미세조정의 핵심 도구: `Trainer` API**

- **교안 7:** 우리는 대용량 데이터를 다루기 위해 `Dataset`과 `DataLoader`를 짰습니다.
    
- **교안 3~5:** 우리는 모델 학습을 위해 `for epoch...`으로 시작하는 **'학습 루프(Training Loop)'**를 직접 짰습니다.
    
    - `optimizer.zero_grad()`
        
    - `loss.backward()`
        
    - `optimizer.step()`
        
- 허깅페이스의 `Trainer`는 이 모든 과정을 **단 세 줄**로 압축해 줍니다.
    
    1. `TrainingArguments` (학습 설정) 정의
        
    2. `Trainer` (모델, 설정, 데이터) 정의
        
    3. `trainer.train()` (학습 시작)
        

---

## 허깅페이스 `Trainer`로 미세조정

- 이 코드는 가상의 데이터로 실제 미세조정 학습을 수행합니다.
    
- **필수 설치:** `pip install transformers datasets accelerate` (accelerate는 Trainer 실행에 필요)
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, # 🌟 '작업용' 모델
    Trainer,                            # 🌟 학습 루프 자동화
    TrainingArguments                   # 🌟 학습 설정
)
from datasets import Dataset # 🌟 허깅페이스 데이터셋 라이브러리

# --- 0. 데이터 준비 (가상의 영화 리뷰) ---
# (실제로는 .csv 파일이나 허깅페이스 Hub에서 바로 로드합니다)
raw_data = {
    "text": [
        "This movie was fantastic!", 
        "I absolutely loved it.",
        "The acting was terrible.",
        "What a waste of time."
    ],
    # 🌟 'label' (정답): 1 (긍정), 0 (부정)
    "label": [1, 1, 0, 0] 
}
# 딕셔너리를 허깅페이스 Dataset 객체로 변환
dataset = Dataset.from_dict(raw_data)

# --- 1. 토크나이저 및 모델 로드 ---
MODEL_NAME = "bert-base-uncased" # (작은 BERT 모델로 테스트)

# 1-1. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 1-2. (중요!) 분류 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2 # 🌟 "이 모델은 2개(0과 1)로 분류하는 모델이야"
)

# --- 2. 데이터 토큰화 ---
# 데이터셋 전체에 토크나이저를 한 번에 적용하는 함수
def tokenize_function(examples):
    # padding, truncation을 동시에 수행
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# dataset.map()을 사용해 'tokenize_function'을 모든 데이터에 적용
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# (학습에 필요한 'text' 열은 제거하고 PyTorch 텐서로 포맷 변경)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# (실제로는 train/validation set을 나누지만 여기서는 단순화를 위해 전체 사용)
train_dataset = tokenized_dataset

# --- 3. 학습(Training) 설정 ---

# 3-1. TrainingArguments: 학습에 필요한 모든 설정을 정의
training_args = TrainingArguments(
    output_dir="./results",      # 모델과 로그가 저장될 폴더
    num_train_epochs=3,          # 총 학습 Epoch
    per_device_train_batch_size=2, # 배치 사이즈
    logging_steps=1,             # 1번 step마다 로그 출력
)

# 3-2. Trainer: 학습을 실행할 주체
trainer = Trainer(
    model=model,                 # 🌟 미세조정할 모델
    args=training_args,          # 🌟 학습 설정
    train_dataset=train_dataset, # 🌟 학습 데이터
    # (실제로는 eval_dataset=... 도 넣어줍니다)
)

# --- 4. 학습 시작! ---
print("--- 미세조정(Fine-Tuning) 시작 ---")
trainer.train()
print("--- 학습 완료 ---")

# --- 5. 학습된 모델로 예측해보기 ---
print("\n--- 학습된 모델로 예측 ---")
test_text = "I really enjoyed this film."
inputs = tokenizer(test_text, return_tensors="pt") # 토큰화

# (미분 계산 안 함)
with torch.no_grad():
    logits = model(**inputs).logits # 모델의 예측 결과 (raw score)

# 로짓(logits) 중 가장 높은 값의 인덱스(0 또는 1)를 찾음
predicted_class_id = torch.argmax(logits, dim=1).item()
print(f"입력: {test_text}")
print(f"예측: {'긍정(1)' if predicted_class_id == 1 else '부정(0)'}")
```