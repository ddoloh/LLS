# LLS
for sharing wiht mr.choi




### 참고사항 
기존에 작동되던 코드에 뒤집어 씌운 상태로 작업하여 코드 백업 어려움.
LLE, SMOTE process를 실행하기 위해서는 전체 데이터셋에 대해서 실행해야 하는데, tensorflow의 경우 batch상태로 입력되기에 model build process안에 집어 넣기 어려움..(single session의 경우)

### 참고 블로그/document

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
