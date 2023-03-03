## Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/rnn-cme-prediction.svg)](https://zenodo.org/badge/latestdoi/432874495)

## Authors

Hao Liu, Chang Liu, Jason T. L. Wang, and Haimin Wang

## Contributors

Yasser Abduallah and Shaya Goldberg

## Abstract

We present two recurrent neural networks (RNNs), one based on gated recurrent units and the other based on long short-term memory, for predicting whether an active region (AR) that produces an M- or X-class flare will also produce a coronal mass ejection (CME). We model data samples in an AR as time series and use the RNNs to capture temporal information of the data samples. Each data sample has 18 physical parameters, or features, derived from photospheric vector magnetic field data taken by the Helioseismic and Magnetic Imager (HMI) on board the Solar Dynamics Observatory (SDO). We survey M- and X-class flares that occurred from 2010 May to 2019 May using the Geostationary Operational Environmental Satellite's X-ray flare catalogs provided by the National Centers for Environmental Information (NCEI), and select those flares with identified ARs in the NCEI catalogs. In addition, we extract the associations of flares and CMEs from the Space Weather Database Of Notifications, Knowledge, Information (DONKI). We use the information gathered above to build the labels (positive versus negative) of the data samples at hand. Experimental results demonstrate the superiority of our RNNs over closely related machine learning methods in predicting the labels of the data samples. We also discuss an extension of our approach to predict a probabilistic estimate of how likely an M- or X-class flare will initiate a CME, with good performance results. To our knowledge this is the first time that RNNs have been used for CME prediction.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### ccsc_CMEpredict.ipynb (Jupyter Notebook for CMEPredict)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/RNN-CME-prediction/HEAD?labpath=ccsc_CMEpredict.ipynb)

Please note that starting Binder might take some time to create and start the image.

For the latest updates of CMEPredict refer to [https://github.com/deepsuncode/RNN-CME-prediction](https://github.com/deepsuncode/RNN-CME-prediction)

## Installation on local machine

Requires `Python==3.6.x` (was tested on 3.6.8)

Run `pip install -r requirements.txt` (recommended), or manually install the following packages and specified versions:

| Library      | Version | Description                    |
|--------------|---------|--------------------------------|
| pandas       | 1.1.5   | Data analysis                  |
| scikit-learn | 0.24.2  | Neural network libraries       |
| matplotlib   | 3.3.4   | Plotting and graphs            |
| h5py         | 2.10.0  | Data storage and management    |
| tensorflow   | 1.12.0  | Neural network libraries       |
| keras        | 2.2.4   | Artificial neural networks API |


## References

Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks. Liu, H., Liu, C., Wang, J. T. L., Wang, H., ApJ., 890:12, 2020  

https://iopscience.iop.org/article/10.3847/1538-4357/ab6850

https://arxiv.org/abs/2002.10953

https://web.njit.edu/~wangj/RNNcme/


