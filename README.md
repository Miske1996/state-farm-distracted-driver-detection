# State farm distracted driver detection Kaggle competition using Fastai Library

We've all been there: a light turns green and the car in front of you doesn't budge. Or, a previously unremarkable vehicle suddenly slows and starts swerving from side-to-side.

When you pass the offending driver, what do you expect to see? You certainly aren't surprised when you spot a driver who is texting, seemingly enraptured by social media, or in a lively hand-held conversation on their phone.

According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year.

State Farm hopes to improve these alarming statistics, and better insure their customers, by testing whether dashboard cameras can automatically detect drivers engaging in distracted behaviors. Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?


## Setting up the Kaggle API token
 
 ```
!pip install -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"miske01","key":"1fb5c9c5be6ead6c58e09d0a5e1526f1"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```
## Downloading the dataset and extracting the zip file

```
!kaggle competitions download -c state-farm-distracted-driver-detection
!mkdir Dataset
!unzip -I 'utf-8' 'state-farm-distracted-driver-detection.zip' -d 'Dataset'
```
## Importing necessary librairies
```
from fastai import *
from fastai.vision import *
from pathlib import Path
import numpy as np
import pandas as pd
```
## Creating the training and testing ImageDataBunch objects
```
train = Path('Dataset/imgs/train')
test = Path('Dataset/imgs/test')

data_train = (ImageList.from_folder(train)
             .split_by_rand_pct(0.2)
             .label_from_folder()
             .transform(get_transforms(do_flip=False),size = 224)
             ).databunch(bs = 32).normalize(imagenet_stats)

data_test = (ImageList.from_folder(test)
            .split_none()
            .label_from_folder()
            .transform(get_transforms(do_flip = False),size=224)
            )
data_test.valid = data_test.train
data_test = data_test.databunch(bs = 32).normalize(imagenet_stats)
```

## Setting up the learner with Resnet34 as the architecture
```
learner = cnn_learner(data_train,models.resnet34,metrics = accuracy)
```
## Preparing the csv submission file
```
learner.data.valid_dl = data_test.valid_dl
preds = learner.get_preds(DatasetType.Valid)

names = [path for path in test.ls()]
labels = pd.DataFrame(names,columns = ['img'])
labels.img = labels.img.astype(str)
labels = labels.img.str.rsplit('/',1,expand=True)
labels.drop(0,axis = 1 ,inplace = True)
labels.rename(columns={1:'img'},inplace = True)
columns = data_train.classes
submission = pd.DataFrame(preds[0].numpy(),columns = columns,index = [labels.img])
submission.reset_index(inplace = True)
submission.to_csv('submission.csv')
```
