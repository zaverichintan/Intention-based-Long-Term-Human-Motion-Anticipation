# junn
deep learning framework build around pytorch for personal projects


## Install

```
pip install git+https://github.com/jutanke/junn.git
```
or locally by
```
python setup.py install
```

## Usage

### Scaffolding
```python
import torch
import torch.nn as nn
from junn.trainer import AbstractTrainer


class MyTrainer(AbstractTrainer):
    def __init__(device, force_new_training=False):
        
        # any sort of parametrization
        inp = 32
        outp = 1
        train_params = {
            "in": inp,
            "out": outp
            "datafiles": ['/path/1', '/path/2']
        }

        ff = nn.Sequential(nn.Linear(inp, outp))
        opt_ff = torch.optim.Adam(params, lr=0.001, amsgrad=True)

        models = {
            "ff": (ff, opt_ff)
        }

        super().__init__(
            models=models,
            train_params=train_params,
            device=device,
            force_new_training=force_new_training
            project_folder="/home/user/output"
        )

    def trainer_name(self):
        return "mytrainer"

    def train_step(self, epoch, Data, models):
        raise NotImplementedError

    def val_step(self, epoch, Data, models):
        raise NotImplementedError

    def on_epoch_end(self, epoch):
        raise NotImplementedError

    def loss_names(self):
        return ["loss"]

# =====================
# U S A G E
# =====================
device = torch.device("cuda")

trainer = MyTrainer(device)

dl_train = DataLoader(...)
dl_test = DataLoader(...)

trainer.run(dl_train, dl_test)
```
