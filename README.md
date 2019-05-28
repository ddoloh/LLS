# LLS
for sharing wiht mr.choi
- - - 
### 참고사항 
&nbsp;&nbsp;&nbsp;&nbsp;기존에 작동되던 코드에 뒤집어 씌운 상태로 작업하여 코드 백업 어려움.  
&nbsp;&nbsp;&nbsp;&nbsp;현재 llesmote.py는 LLE process를 단순히 numpy에서 tensorflow package로 치환한 것임.(테스트 과정에서 몇 가지 함수를 계속 바꿔가면서 하여 작동되지 않을 것임..)  
&nbsp;&nbsp;&nbsp;&nbsp;참고로 LLE, SMOTE process를 실행하기 위해서는 전체 데이터셋에 대해서 실행해야 하는데, tensorflow의 경우 batch상태로 입력되기에 전체 데이터셋에 대해 처리하기 어려우며, 구현된 패키지들은 numpy array들을 인풋으로 받는 패키지들이라 model build process안에 집어 넣기 어려움.(single session의 경우)      
&nbsp;&nbsp;&nbsp;&nbsp;multi session의 경우 가능할 것 같음... 위에서 올린 프로젝트를 예로 들면, GURcell-> LLE-SMOTE -> SVM 순으로 작성하는 것이 중단 실험 모델인데, 기존 GRU에서 계산된 'Wx_plus_b'(weight matmul, plus bias) 결과를 LLE-SMOTE process를 거치는 과정에서 에러가 발생됨. 멀티 세션 혹은 부분적으로 학습시키는 것이 현실적일듯.  
&nbsp;&nbsp;&nbsp;&nbsp;또, 이전에 회의하던 중 캐치못했었던 문제가 하나 더 있음. SMOTE process를 중간에 넣게 되면 모델에 들어가는 input의 label은 binary로 입력되어 SMOTE 또한 binary label에 대해서 밖에 실행될 수 밖에 없음.(기존 proposal에 매우 동떨어짐..) 이 문제는 모델을 20-classes(진 10, 가 10)로 분류하면 해결될 듯 함.(다만, 성능은 꽤나 떨어지지 않을까 생각됨..)  
- - - 
### 참고 블로그/document


- - -
### 구조
>gru-svm
>
>>models
>>
>>>gru_svm.py
>>
>>>gru_svm_llesmote.py
>>
>>>llesmote.py
>
>>LBP_LLE_SMOTE_.py
>>
>>gru_svm_main.py
>>
>>play.sh
