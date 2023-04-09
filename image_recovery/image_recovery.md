## 프로젝트 1. 경사 하강법으로 이미지 복원하기

### 프로젝트 개요와 목표

이번 프로젝트에서 우리가 풀 문제는 다음과 같습니다.

이미지 처리를 위해 만들어 두었던 weird_function() 함수에 실수로 버그가 들어가 100×100 픽셀의 오염된 미미지가 만들어졌습니다. 이 오염된 이미지와 오염되기 전 원본 이미지를 동시에 파일로 저장하려고 했으나, 모종의 이유로 원본 이미지 파일은 삭제된 상황입니다. 다행히도 weird_function()의 소스코드는 남아 있습니다. 오염된 이미지와 weird_function()을 활용해 원본 이미지를 복원해봅시다.

*참고자료: https://github.com/jcjohnson/pytorch-examples, NYU Intro2ML*


```python
import torch
import pickle
import matplotlib.pyplot as plt
```


```python
shp_original_img = (100, 100)
broken_image =  torch.FloatTensor( pickle.load(open('./broken_image_t.p', 'rb'),encoding='latin1' ) )
```


```python
plt.imshow(broken_image.view(100,100)) 
```




    <matplotlib.image.AxesImage at 0x7fb2878b0ac8>




    
![png](output_3_1.png)
    



```python
def weird_function(x, n_iter=5):
    h = x    
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat( (zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0 )
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0:
            h = torch.cat( (h[h.shape[0]//2:],h[:h.shape[0]//2]), 0  )
    return h
```


```python
def distance_loss(hypothesis, broken_image):    
    return torch.dist(hypothesis, broken_image)
```


```python
random_tensor = torch.randn(10000, dtype = torch.float)
```


```python
lr = 0.8
for i in range(0,20000):
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad():
        random_tensor = random_tensor - lr*random_tensor.grad
    if i % 1000 == 0:
        print('Loss at {} = {}'.format(i, loss.item()))
```

    Loss at 0 = 12.468249320983887
    Loss at 1000 = 1.2029341459274292
    Loss at 2000 = 0.5675680041313171
    Loss at 3000 = 0.3876642882823944
    Loss at 4000 = 0.3058801293373108
    Loss at 5000 = 0.2560693621635437
    Loss at 6000 = 0.2202509492635727
    Loss at 7000 = 0.1919025480747223
    Loss at 8000 = 0.168051615357399
    Loss at 9000 = 0.1470954418182373
    Loss at 10000 = 0.12807640433311462
    Loss at 11000 = 0.11039573699235916
    Loss at 12000 = 0.09367085248231888
    Loss at 13000 = 0.07765314728021622
    Loss at 14000 = 0.06217623129487038
    Loss at 15000 = 0.047126807272434235
    Loss at 16000 = 0.03242645412683487
    Loss at 17000 = 0.0211363285779953
    Loss at 18000 = 0.021164026111364365
    Loss at 19000 = 0.021166549995541573
    


```python
plt.imshow(random_tensor.view(100,100).data)
```




    <matplotlib.image.AxesImage at 0x7fb287404d68>




    
![png](output_8_1.png)
    

