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
from junn.scaffolding import Scaffolding


class MyModel(Scaffolding):

    def __init__(self, model_seed=0,
                 force_new_training=False):
        """
        :param model_seed: {int} random seed for training. Allows for training the same model
                        with different seeds
        :param force_new_training: {boolean} if True delete existing training data
        """
        super().__init__(force_new_training=force_new_training,
                         model_seed=model_seed)
         
        self.model = torch....

    def get_unique_directory(self):
        """ return {string} that uniquely identifies this model. The model seed will be added
            to the string automatically and does not need to be added in this function
        """
        return "mymodel"
    
    def forward(self, x):
        """ pytorch forward function
        """
        return self.model(x)
```

### Trainer
```python
import torch
from junn.training import Trainer
from torch.utils.data import DataLoader

class MyTrainer(Trainer):

    def __init__(self, device, 
                 force_new_training=False, model_seed=0):
        """
        :param device: {torch::device}
        :param force_new_training: {boolean} see Scaffolding
        :param model_seed: {int} see Scaffolding
        """

        model = MyModel(model_seed, force_new_training)
        
        # put_all_models_in_common_dir: if the trainer supervises more than
        # one model the training files can be summarized into a single
        # directory by setting this variable to "True"
        super().__init__([model], device=device,
                         force_new_training=force_new_training,
                         put_all_models_in_common_dir=False)
    
    def get_folder_name(self):
        """ This function is required if {put_all_models_in_common_dir==True}
        """ 
        return "mytrainer"

    def loss_names(self):
        """ Defines the names of the losses. This list has to coincide with the return
            losses from train and validate step
        """
        return ['loss', ' pose']
    
    def train_step(self, epoch, Data, optim):
        """ ensure to return the correct number of losses
        """
        pass
    
    def val_step(self, epoch, Data):
        """ ensure to return the correct number of losses
        """
        pass


# =====================
# U S A G E
# =====================
device = torch.device("cuda")

trainer = MyTrainer(device)

dl_train = DataLoader(...)
dl_test = DataLoader(...)

optim = torch.optim.Adam(params, lr=0.001, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.96)

trainer.run(dl_train, dl_test, optim=optim, optim_scheduler=scheduler)
```
