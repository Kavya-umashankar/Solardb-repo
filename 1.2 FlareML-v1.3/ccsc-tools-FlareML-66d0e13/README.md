## Predicting Solar Flares with Machine Learning<br>
[![DOI](https://zenodo.org/badge/417927505.svg)](https://zenodo.org/badge/latestdoi/417927505)

## Authors
Yasser Abduallah, Jason T. L. Wang, Haimin Wang

## Abstract

Solar flare prediction plays an important role in understanding and forecasting space weather.
The main goal of the Helioseismic and Magnetic Imager (HMI), one of the instruments on
NASA&#39;s Solar Dynamics Observatory, is to study the origin of solar variability and characterize
the Sun&#39;s magnetic activity. HMI provides continuous full-disk observations of the solar vector
magnetic field with high cadence data that lead to reliable predictive capability; yet, solar flare
prediction effort utilizing these data is still limited. Here we present a flare prediction system,
named FlareML, for predicting solar flares using machine learning (ML) based on HMI’s data
products. Specifically, we construct training data by utilizing the physical parameters provided
by the Space-weather HMI Active Region Patches (SHARP) and categorize solar flares into four
classes, namely B, C, M, X, according to the X-ray flare catalogs available at the National
Centers for Environmental Information (NCEI). Thus, the solar flare prediction problem at hand
is essentially a multi-class (i.e., four-class) classification problem. The FlareML system employs
four machine learning methods to tackle this multi-class prediction problem. These four methods
are: (i) ensemble (ENS), (ii) random forests (RF), (iii) multilayer perceptrons (MLP), and (iv)
extreme learning machines (ELM). ENS works by taking the majority vote of the results
obtained from RF, MLP and ELM.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### ccsc_FlareML.ipynb (Jupyter Notebook for FlareML)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/FlareML/HEAD?labpath=ccsc_FlareML.ipynb)

Please note that starting Binder might take some time to create and start the image.


For the latest updates of FlareML refer to https://github.com/deepsuncode/Machine-learning-as-a-service

## Installation on local machine

|Library | Version   | Description  |
|---|---|---|
|matplotlib|3.4.2| Graphics and visualization|
|numpy| 1.19.5| Array manipulation|
|scikit-learn| 0.24.2| Machine learning|
| sklearn-extensions | 0.0.2  | Extension for scikit-learn |
| pandas|1.2.4| Data loading and manipulation|
