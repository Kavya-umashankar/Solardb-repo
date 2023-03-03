## A Transformer-Based Framework for Geomagnetic Activity Prediction<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/kp.svg)](https://doi.org/10.5281/zenodo.7047116)


## Authors
Yasser Abduallah, Jason T. L. Wang, Chunhui Xu, and Haimin Wang

## Abstract

Geomagnetic activities have a crucial impact on Earth, which can affect spacecraft and electrical power grids.
Geospace scientists use a geomagnetic index,
called the Kp index,
to describe the overall level of geomagnetic activity.
This index is an important indicator of disturbances in the Earth's magnetic field 
and is used by the U.S. Space Weather Prediction Center as an alert and warning service
for users who may be affected by the disturbances.
Early and accurate prediction of the Kp index is essential for 
preparedness and disaster risk management. 
In this paper, we present a novel deep learning method, named KpNet, 
to perform short-term, 1-9 hour ahead, forecasting of the Kp index 
based on the solar wind parameters taken from the NASA Space Science Data Coordinated Archive. 
KpNet combines transformer encoder blocks with Bayesian inference, 
which is capable of quantifying both aleatoric uncertainty (data uncertainty) and 
epistemic uncertainty (model uncertainty)
when making Kp predictions. 
Experimental results show that KpNet outperforms closely related machine learning methods 
in terms of the root mean square error and R-squared score. 
Furthermore, KpNet can provide both data and model uncertainty quantification results, which the existing methods cannot offer.
To our knowledge, this is the first time that
Bayesian transformers have been used for Kp prediction.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### YA_01_ATransformerBasedFrameworkForGeomagneticActivity.ipynb (Jupyter Notebook for Kp-prediction)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/Kp-prediction/HEAD?labpath=YA_01_ATransformerBasedFrameworkForGeomagneticActivity.ipynb)

Please note that starting Binder might take some time to create and start the image.

Please also note that the execution time in Binder varies based on the availability of resources. The average time to run the notebook is 30-40 minutes, but it could be more.

For the latest updates of the tool refer to https://github.com/deepsuncode/Kp-prediction

## Installation on local machine
|Library | Version   | Description  |
|---|---|---|
|keras| 2.8.0 | Deep learning API|
|numpy| 1.22.4| Array manipulation|
|scikit-learn| 1.0.2| Machine learning|
|sklearn| latest| Tools for predictive data analysis|
|matlabplot| 3.5.1| Visutalization tool|
| pandas|1.4.1| Data loading and manipulation|
| seaborn | 0.11.2| Visualization tool|
| scipy | 1.8.1| Provides algorithms for optimization and statistics|
| tensorflow| 2.8.0| Comprehensive, flexible ecosystem of tools and libraries for machine learning |
| tensorflow-gpu| 2.8.0| Deep learning tool for high performance computation |
|tensorflow-probability | 0.14.1| For probabilistic models|
