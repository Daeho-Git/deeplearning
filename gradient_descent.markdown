# deeplearning - first ANN with penguinbro

> 경사하강법으로 이미지 복원 코드체크

```python
import torch
import pickle
import matplotlib.pyplot as plt

broken_image = torch.FloatTenseor( pickle.load(open('./broken_image_t.p', 'rb'),encoding='latin1'))

plt.imshow(broken_image.view(100,100))
```

## 코드 분석
---
torch.FloatTensor()은 파이토치에서 32-bit floating point 데이터 타입을 가지는 tensor를 생성하는 함수입니다.

pickle 모듈은 파이썬 객체를 직렬화하여 바이트 형태로 저장하거나, 반대로 직렬화된 바이트를 객체로 역직렬화하는 기능을 제공합니다.
 open() 함수를 사용하여 './broken_image_t.p' 경로에 있는 파일을 바이트 형태로 열고, 'rb' 모드를 사용하여 읽기 모드로 열어줍니다. 
encoding 옵션은 pickle 데이터가 바이트 형태로 저장된 경우 지정하는 인코딩 방식입니다.
따라서 pickle.load() 함수를 사용하여 pickle로 저장된 데이터를 읽어와서 Tensor로 변환합니다.
 torch.FloatTensor() 함수를 사용하여 파이토치 Tensor 형태로 변환한 이유는,
 이미지 처리와 같은 딥러닝 모델 학습에서 데이터를 빠르게 처리하고 효율적으로 계산하기 위해서는 
Tensor 형태로 데이터를 변환하여 사용하는 것이 좋기 때문입니다.

- r: 읽기 모드
- w: 쓰기 모드
- a: 추가 모드
- rb: 바이너리 읽기 모드
- wb: 바이너리 쓰기 모드
- ab: 바이너리 추가 모드
- r+: 읽기/쓰기 모드 (파일 처음 위치에서 열기)
- w+: 읽기/쓰기 모드 (파일을 새로 생성하거나 덮어쓰기)
- a+: 읽기/추가 모드 (파일 끝에서 열기)

