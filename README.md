# VicReg
## A VicReg Implementation in pytorch [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)

![VicReg architecture image from the paper](https://github.com/markpesic/VicReg/blob/master/images/vicreg.png?raw=true)

## Model
```python
from Vic.Vicreg import VicReg

model = VicReg(
    backend='resnet50',
    input_size=2048,
    output_size=256,
    depth_projector=3,
    pretrained_backend=False)
   
```

## Training
```python
import torch
from Vic.Vicreg import VicReg
from Vic.utils import criterion, get_byol_transforms
#train_loader, size, mean, std, lr and device given by the users

t, t1, _ = get_byol_transforms(size, mean, std)

model = VicReg(
    backend='resnet50',
    input_size=2048,
    output_size=256,
    depth_projector=3,
    pretrained_backend=False)
    
model = model.to(device)
optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum= 0.9, weight_decay=1.5e-4)

for epoch in range(30):
    model.train()
    for batch, _ in train_loader:
       batch = batch.to(device)
       x = t(batch)
       x1 = t1(batch)

       fx = model(x)
       fx1 = model(x1)
       loss = criterion(fx, fx1)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

```

## Citation
```bibtex
@misc{https://doi.org/10.48550/arxiv.2105.04906,
  doi = {10.48550/ARXIV.2105.04906},
  url = {https://arxiv.org/abs/2105.04906},
  author = {Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
