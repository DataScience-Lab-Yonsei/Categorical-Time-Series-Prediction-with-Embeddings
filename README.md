# Categorical-Time-Series-Prediction-with-Embeddings
![image](https://user-images.githubusercontent.com/87710236/163808952-e3c93f9e-2f47-4373-b624-bde02608babf.png)

플랫폼 내에서 고객 행동 데이터를 기반으로 다음 행동을 예측하여 Personalized한 서비스 제공할 수 있는 기회 제공을 목적으로 하며, L.POINT 데이터를 활용하였다.
### DSL 모델링 프로젝트 (2022. 04. 07.)
___

### Collaborators
김지오, 박준우, 이승재, 장윤태, 조수연, 조영규
___
## File Definition
### 📁code
* #### Embedding_action_type_prediction   
Embedding을 적용한 후, action_type을 예측하는 모델링 코드이다.(모델 튜닝 전)   
* #### Transformer_action_type_prediction   
직관적인 분류, 임베딩 기반 분류로 action_type을 군집화하여 예측한 모델링 코드이다.
* #### LSTM_action_type_prediction   
LSTM으로 action_type을 예측하는 모델링 코드이다. Embedding&Transformer기법을 활용한 모델보다 성능이 낮아 summary에는 활용되지 않았다.
* #### time_series_clustering_10series & time_sereis_clustering_2   
시계열 클러스터링 기법 (kmeans, kshape 등)을 사용하여 군집을 변수로 만들어 action type을 예측한 코드이다. 군집화변수를 넣은 것이 오히려 성능이 낮아져 시계열 클러스터링이 잘 되지 않는 것이 데이터의 한계였다.

### 📁data
제 6회 L.POINT BIG DATA Competition에서 제공한 데이터이다. 자세한 데이터에 대한 설명은 reference '제 6회 L.POINT Big Data Competition-설명회 자료'를 참고하면 된다.   
* #### 거래정보.csv   
온라인 또는 오프라인에서 구매한 내역이 담겨있다.
* #### 고객정보.csv
고객 Demographic 데이터(성별, 연령대)이다.   
* #### 상품분류정보.csv   
상품에 대한 정보가 담겨있다.
* #### 온라인행동정보.csv   
고객의 온라인 행동에 대한 기록으로써, 유입부터 구매까지 모든 행동 과정이 담겨있다.
___
# modeling_summary를 통한 프로젝트 설명

# Overview
![image](https://user-images.githubusercontent.com/87710236/163808879-1f22bf51-ec87-4327-8676-11a8e7861380.png)

# Data Preprocessing
### Sliding window
'Sliding window'기법을 이용하여 '온라인행동정보.csv' 데이터에서 CLNT_ID(고객), SESS_ID(세션) 별 HIT_SEQ(조회일련번호)에 따라 정렬한 뒤, action_types데이터를 앞의 10 steps를 x, 11번째 step을 y로 정한다. window를 1 step씩 뒤로 이동시키며 이 과정을 반복하여 데이터를 생성하였다.
![image](https://user-images.githubusercontent.com/87710236/163809130-bd529dc9-9d33-4861-b4b9-24eac47dfa9f.png)


# Model Architecture
BST(Behavior Sequence Transformer) 모델을 참고했다.(Refrence 첨부) 모델의 자세한 구조는 아래와 같다.
![image](https://user-images.githubusercontent.com/87710236/163809567-e47feb71-be39-407f-b795-2c2ef6fe6063.png)

![image](https://user-images.githubusercontent.com/87710236/163809514-ea51ded8-89bf-4cab-83f3-9172f715bcdf.png)

# Time Series Application of Categorical Data
주로 연속적인 데이터를 예측하는 시계열 예측모델링을 Categorical Data에 적용하기 위해 아래와 같은 기법을 적용하였다.

## 1) One hot encoding
![image](https://user-images.githubusercontent.com/87710236/163809728-b217ebef-5b80-4884-870a-ffe31c95b0bb.png)
* 특징   
✔ 카테고리 개수만큼 차원을 갖는 벡터 생성   
* 단점   
✔ 카테고리 개수가 커지면 데이터가 sparse 해짐   
✔ 카테고리 벡터 간에 거리가 동일하므로 관계 분석에 어려움   

## 2)Embedding
![image](https://user-images.githubusercontent.com/87710236/163809839-c3c1860e-d621-4eb4-b5f2-936e025df2b8.png)
* 특징   
✔ 각 카테고리에 대해 원하는 차원수(D)만큼 설정할 수 있음 (sparse 방지)   
✔ Embedding 값이 학습을 통해 의미적으로 비슷한, 즉, 유의미한 관계를 분석하여 데이터를 군집으로 해석할 수 있는 효과가 있음   

# Result
![image](https://user-images.githubusercontent.com/87710236/163810941-152be157-c434-4fee-aa21-410378cd84b2.png)
![image](https://user-images.githubusercontent.com/87710236/163810987-984be1e8-4ab2-48b6-a3a4-20054f483f3d.png)

# Model tuning
![image](https://user-images.githubusercontent.com/87710236/163810830-00ce737b-fafe-401b-bab5-60b8d1b2a6e2.png)
![image](https://user-images.githubusercontent.com/87710236/163810862-ca98b952-9bf4-4a3d-bbe9-82d0ac3828fc.png)

# Result
![image](https://user-images.githubusercontent.com/87710236/163810898-bae74eee-7760-4ead-bdf6-0ec994a60968.png)

# Conclusion
![image](https://user-images.githubusercontent.com/87710236/163810768-29b4242e-0c46-4db2-a9d5-5a68f9fd83d0.png)

# Reference
- BST(Behavior Sequence Transformer) for E-commerce Recommendation in Alibaba- Qiwei Chen, Huan Zhao∗ Wei Li, Pipei Huang, Wenwu Ou Alibaba Group Beijing & Hangzhou, China   
- 제6회 L.POINT Big Data Competition-설명회 자료
- https://arxiv.org/pdf/2106.01490.pdf
- https://dacon.io/codeshare/1625
