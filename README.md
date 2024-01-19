## Organisation

  - A: includes Task A code ```TaskA.py```, all classes and functions required are already in ```TaskA.py```
      - ```TaskA_ResNet.ipynb```: can be run in Jupyter Notebook
      - ```Trained_Model.ipynb ```: file that can run the pretrained model
      - ```checkpoint_resnet100.pth```: pretrained model with 100 epochs
      - ```loss.png```: train loss curve used in the preort
   
   - B: includes Task A code ```TaskB.py```, all classes and functions required are already in ```TaskB.py```
      - ```TaskB_ResNet.ipynb```: can be run in Jupyter Notebook
      - ```Trained_Model.ipynb ```: file that can run the pretrained model
      - ```checkpoint_resnet100.pth```: pretrained model with 100 epochs
      - ```loss.png```: train loss curve used in the preort
    
  - Dataset: includes directories of ```PathMNIST``` and ```PneumoniaMNIST```. Bith directories have a ```temp.txt``` in it. ```pneumoniamnist.npz``` and ```pathmnist.npz``` can be put into corresponding directory.
  
  - ```main.py```: used for runing ```TaskA.py``` and ```TaskB.py```


## Package required
```
torch==2.1.1+cu118
torchvision==0.16.1+cu118
medmnist
sklearn
scipy
numpy
matplotlib
tqdm
```

  To install ```medmnist``` as a standard Python package, use ```pip```

        pip install medmnist

  Or install from sorce
  
        pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

## Note
  - The default training epoch is ```10```. It can also be changed in ```TaskA.py``` and ```TaskB.py``` with the parameter name of ```num_epoches```
    
