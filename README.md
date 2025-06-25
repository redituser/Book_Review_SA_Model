---

# Book\_Review\_SA - 학습 데이터 & 모델 비교

---

## 1. 모델 학습 방법

### 1.1 기존 라벨링 방식 (KNU 감성사전)

* KNU 감성사전을 기반으로 단어 매칭으로 감성 점수를 계산
* 문장 내 단어 점수를 합산하여 감성 지수 도출

예시 코드:

```python
score_dict = { '좋아요':1 , '최고에요':1, '훌륭해요':1, '멋져요':1 , 
               '별로예요':-1, '싫어요':-1, '나빠요':-1, '비싸요':-1 }
str_review = '그 영화는 훌륭해요 멋져요 그래서 비싸요'

def s_sentiment(sentence):
    sentence = sentence.split(' ')
    all_score = 0
    for word, value in score_dict.items():
        for i in sentence:
            if i == word:
                all_score += value
    print(all_score)

s_sentiment(str_review)
```

## KNU 라벨링 기준 예시
![KNU 예시](https://github.com/user-attachments/assets/a9bb2f5e-49b5-4604-99e5-9e98e29fc8a2)
#### 한계: 단순 단어 매칭 기반이므로 정확도가 낮음
---

### 1.1 개선 라벨링 방식 (GEMMA 기반 분류기)

* GEMMA를 활용하여 리뷰를 `긍정`, `중립`, `부정`으로 직접 분류
* 결과를 `.csv`로 저장하여 학습에 활용

 GEMMA 결과 예시
![GEMMA 예시](https://github.com/user-attachments/assets/6bf7a376-1e8d-4516-8b96-11638ba2568a)

---

### 1.2 데이터 전처리

#### 클래스 매핑 함수:

```python
def map_sentiment(score):
    if score < 0: return 0   # 부정
    elif score == 0: return 1 # 중립
    else: return 2           # 긍정
```

#### 토크나이저 비교:

| 방식    | 사용 모델                         | 설명              |
| ----- | ----------------------------- | --------------- |
| 기존 방식 | `Tokenizer` + `pad_sequences` | 텍스트를 정수 시퀀스로 변환 |
| 개선 방식 | `klue/bert-base` (BERT 토크나이저) | 문맥 기반 임베딩 사용    |

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
encoded_inputs = tokenizer(reviews, max_length=128, padding=True, truncation=True, return_tensors='tf')
```

---

### 1.3 모델 구조 비교

#### (기존) BiLSTM 모델 구조:

```python
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```

#### (개선) BERT 기반 모델 구조:

```python
class CustomBertForSequenceClassification(tf.keras.Model):
    def __init__(self, bert_model_core, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model_core
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_labels, name="classifier")

    def call(self, inputs, training=False):
        outputs = self.bert(inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            training=training)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits
```

---

## 2. 모델 성능 비교

### 데이터 크기

> **총 리뷰 수: 304,027건**

---

### BiLSTM 결과

* **테스트 정확도**: `77.80%`

- **정확도 그래프**  
  ![BiLSTM 정확도](https://github.com/user-attachments/assets/0f07b491-83b3-46d8-b16a-f480b4d0fe72)


- **Confusion Matrix**  
  ![Confusion](https://github.com/user-attachments/assets/9d83b549-a680-4978-935a-5a629fc0a46e)

- **Classification Report**  
  ![Classification Report](https://github.com/user-attachments/assets/5a2165ce-6e8b-4f8a-988c-de97d14e9ce3)

---

### BERT 결과

* **테스트 정확도**: `87.34%`

- **정확도 그래프**  
  ![BERT 정확도](https://github.com/user-attachments/assets/12eb0068-e8b9-4a1a-9b0c-04a4128081ff)

- **Confusion Matrix**  
  ![Confusion](https://github.com/user-attachments/assets/5dd0e680-4557-4491-b1ec-3c085b23957f)

- **Classification Report**  
  ![Classification Report](https://github.com/user-attachments/assets/805a62b9-cbd8-48b5-a99f-6e2e9fff603b)

---

| 비교 항목             | BiLSTM     | BERT                    |
| ----------------- | ---------- | ----------------------- |
| 테스트 정확도           | 77.80%     | **87.34%**              |
| Weighted F1-score | 0.7639     | **0.87**                |
| 클래스 불균형 대응력       | 낮음 (편향 존재) | 상대적으로 우수                |
| 문맥 이해             | 제한적        | **우수 (Transformer 기반)** |
| 처리 속도             | 빠름         | 상대적으로 느림                |
| 모델 크기/무게          | 가벼움        | 큼 (사전학습 파라미터 포함)        |


>  **결론**:
> BiLSTM은 속도와 단순 구조에서는 유리하지만, 정확도 및 F1-score 측면에서 부족함
> BERT는 정밀한 분류가 필요한 실서비스에서는 훨씬 신뢰도가 높음
