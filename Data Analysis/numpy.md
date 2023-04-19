## NUMPY
---
>NumPy는 파이썬의 대표적인 수치 계산 라이브러리로, 다차원 배열과 벡터 연산에 최적화된 기능을 제공합니다. NumPy의 기본 사용법과 예제를 알아보겠습니다.
---

1. 라이브러리 설치 및 가져오기

먼저 NumPy를 설치합니다.

```bash
pip install numpy
```
그리고 파이썬 코드에서 NumPy를 가져옵니다.

```python
import numpy as np
```


2. 배열 생성

NumPy에서 배열을 생성하는 방법은 다양합니다. 다음은 몇 가지 예입니다.

```python
# 파이썬 리스트로부터 배열 생성
arr = np.array([1, 2, 3, 4])
print("1차원 배열:", arr)

# 영행렬 생성
zeros = np.zeros((3, 3))
print("영행렬:\n", zeros)

# 일행렬 생성
ones = np.ones((3, 3))
print("일행렬:\n", ones)

# 단위행렬 생성
identity = np.eye(3)
print("단위행렬:\n", identity)

# 등간격 배열 생성
linspace = np.linspace(0, 1, 5)
print("linspace 배열:", linspace)

# 랜덤 배열 생성
random_arr = np.random.rand(3, 3)
print("랜덤 배열:\n", random_arr)
```


3. 배열 인덱싱 및 슬라이싱

NumPy 배열의 특정 요소에 접근하거나, 배열의 일부분을 추출할 수 있습니다.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 인덱싱
print("인덱싱:", arr[1, 1])  # 출력 결과: 5

# 슬라이싱
print("슬라이싱:\n", arr[0:2, 1:3])
```


4. 배열 연산

NumPy는 배열 간의 다양한 연산을 지원합니다.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 요소별 덧셈
print("요소별 덧셈:\n", A + B)

# 요소별 곱셈
print("요소별 곱셈:\n", A * B)

# 행렬곱
print("행렬곱:\n", np.dot(A, B))
```


5. 기타 유용한 함수

NumPy에는 다양한 유용한 함수가 있습니다.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 배열 전치
print("전치:\n", arr.T)

# 배열 차원 변경
print("차원 변경:\n", arr.reshape(1, 9))

# 배열 결합 (수직 방향)
vstack_arr = np.vstack((arr, arr))
print("수직 결합:\n", vstack_arr)

# 배열 결합 (수평 방향)
hstack_arr = np.hstack((arr, arr))
print("수평 결합:\n", hstack_arr)

# 배열 분할 (수직 방향)
upper, lower = np.vsplit(arr, [2])
print("수직 분할 (상단):\n", upper)
print("수직 분할 (하단):\n", lower)

# 배열 분할 (수평 방향)
left, right = np.hsplit(arr, [2])
print("수평 분할 (좌측):\n", left)
print("수평 분할 (우측):\n", right)

# 최댓값 및 최솟값
print("최댓값:", arr.max())
print("최솟값:", arr.min())

# 평균, 분산, 표준편차
print("평균:", arr.mean())
print("분산:", arr.var())
print("표준편차:", arr.std())

# 합계 및 곱셈
print("합계:", arr.sum())
print("곱셈:", arr.prod())
```
