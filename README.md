# Precipitation forecast post-processing and ensemble member generation using latent diffusion model

This project aims to bias correct and downscale Global Forecast System (GFS) precipitation forecasts in the Conterminous United States (CONUS) by using Latent Diffusion Model (LDM). The post-processing is probabilistic, it generates ensemble members from the given GFS determinstic forecasts.

## Data
* The forecast to post-process:
  * GFS 3 hourly accumulated total precipitation (APCP) up to 36 hours
* Predictors:
  * APCP
  * Convective Available Potential Energy (CAPE)
  * Total column precipitable water (PWAT)
  * 800 hPa air temperature, u, v, and relative humidity
  * Elevation, forecast lead time
* Learning target:
  * Multi-Radar Multi-Sensor (MRMS) hourly and 3 hourly quantitative precipitation estimation

## Method
The project containts three neural networks: 

* A Vector Quantisation Variational Autoencoder (VQ-VAE) that projects MRMS data to a regularized latent space
* An autoencoder that embeds GFS predcitors, elevation, and forecast lead times as feature vectors
* A diffusion model that applies embedded GFS as conditional inputs and generate MRMS-like outputs in the latent space. 

## Note
The project is in its early stage.

