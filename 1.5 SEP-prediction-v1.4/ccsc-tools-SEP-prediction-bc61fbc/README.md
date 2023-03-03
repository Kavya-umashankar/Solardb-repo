## Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/sep.svg)](https://zenodo.org/record/6730145#.YrZxAXbMJD8)


## Authors
Yasser Abduallah, Vania K. Jordanova, Hao Liu, Qin Li1, Jason T. L. Wang, and Haimin Wang

## Abstract

Solar energetic particles (SEPs) are an essential source of space radiation, and are hazardous for humans in space,
spacecraft, and technology in general. In this paper, we propose a deep-learning method, specifically a bidirectional
long short-term memory (biLSTM) network, to predict if an active region (AR) would produce an SEP event given
that (i) the AR will produce an M- or X-class flare and a coronal mass ejection (CME) associated with the flare, or
(ii) the AR will produce an M- or X-class flare regardless of whether or not the flare is associated with a CME. The
data samples used in this study are collected from the Geostationary Operational Environmental Satelliteʼs X-ray
flare catalogs provided by the National Centers for Environmental Information. We select M- and X-class flares
with identified ARs in the catalogs for the period between 2010 and 2021, and find the associations of flares,
CMEs, and SEPs in the Space Weather Database of Notifications, Knowledge, Information during the same period.
Each data sample contains physical parameters collected from the Helioseismic and Magnetic Imager on board the
Solar Dynamics Observatory. Experimental results based on different performance metrics demonstrate that the
proposed biLSTM network is better than related machine-learning algorithms for the two SEP prediction tasks
studied here. We also discuss extensions of our approach for probabilistic forecasting and calibration with
empirical evaluation.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### YA_01_PredictingSEPUsingBiLSTM.ipynb (Jupyter Notebook for SEP-prediction)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/SEP-prediction/HEAD?labpath=YA_01_PredictingSEPUsingBiLSTM.ipynb)

Please note that starting Binder might take some time to create and start the image.

Please also note that the execution time in Binder varies based on the availability of resources. The average time to run the notebook is 10-15 minutes, but it could be more.

For the latest updates of the tool refer to https://github.com/deepsuncode/SEP-prediction

## Installation on local machine

|Library | Version   | Description  |
|---|---|---|
|keras| 2.6.0 | Deep learning API|
|numpy| 1.19.5| Array manipulation|
|scikit-learn| 1.0.1| Machine learning|
| pandas|1.4.1| Data loading and manipulation|
| tensorboard| 2.8.0| Provides the visualization and tooling needed for machine learning|
| tensorflow-gpu| 2.6.0| Deep learning tool for high performance computation |
