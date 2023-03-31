# newspaper-front-page-recognition
## Introduction
Recognizing front pages of historical newspapers. The model operates to output 0 (non-front page) or 1 (front page) on an input image (which is supposed to be an historical newspaper page). The newspapers that are used for training are part of the [*BelgicaPress*](https://www.kbr.be/en/belgica-press/) collection which is hosted and published by the [**Royal Library of Belgium (KBR)**](https://www.kbr.be/en/). The language used in these newspapers are mainly French and Dutch. 
## Backbone
The backbone of the model is [ResNeSt](https://github.com/zhanghang1989/ResNeSt). We used a pretrained version *resnest50_fast_1s4x24d*, please refer to the orignial publication of the authors for further details as well as access to pretrained models. The ending FC layer of the orignial ResNeSt is replaced with a convolutional layer, before further connecting to a classifer (simple combination of one convolutional layer and one FC layer). The code is written on top of the framework published with [CyclGAN](https://github.com/junyanz/CycleGAN). 
## Training and Perforamnce
### Version 0 (current version)
The model is trained using 8 different newspaper titles in the year of 1923. For each title 8 months are chosen randomly for training while the remaining 2 months are used for testing. The accuracy of the model (in terms of bAccuracy and F1) is above 99%.
