# DESIDR10 Photometric redshift estimation with Deep Learning

Updated on 2024-09-20
## Main Work
Apply Deep Learning method in DESI Image Legacy Survey DR10 dataset.

## Current Status(2024-08-21)
Replicate Pasquet et al. (2019) CNN model and use our dataset to check the performance.
Focus on one special objects type, ELG (Emission Line Galaxy). Can't get an acceptable result when using fullsize samples but get a good result when only using z<0.5 samples.
Still seek for improvement -- Trying Another Network Framework to combine catalog data.

## Dir Structure
```
- data/: dataset
- output/: output results including z_prediction, pdf and visualization images.
- weights/: model weights
- logs/: training and testing logs
- analysis.py: analysis the predicted results.
- metrics.py: compute metrices to evaluate algorithm performance
- model.py: define CNN model.
- predict.py: for z prediction.
- train.py: model train and test
- utils.py: related functions.
```
## References:
- **Reference paper**

Pasquet et al. 2019 (https://arxiv.org/abs/1806.06607)
- **Reference repositories**
    - https://github.com/jpasquet/Photoz.git
    - https://github.com/TasosTheodoropoulos/Photoz_SDSS.git





