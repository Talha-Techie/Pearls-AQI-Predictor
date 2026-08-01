# AQI Feature Store Architecture

## Purpose

The feature store provides a consistent interface between the AQI feature pipeline, model training pipeline, and online prediction service.

## Architecture

External Weather/AQI APIs
        |
        v
Feature Collection
        |
        v
Validation
        |
        v
Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Historical Features       Latest Features
        |                      |
        v                      v
Parquet Offline Store    SQLite Online Store
        |                      |
        +----------+-----------+
                   |
                   v
              Feast Registry
                   |
          +--------+--------+
          |                 |
          v                 v
     ML Training       Prediction API

## Offline Store

Historical engineered features are stored in Parquet files and used for training and historical feature retrieval.

## Online Store

SQLite is used as the local online feature store for the internship implementation.

## Leakage Prevention

The following forecast labels are not included in the online Feature View:

- target_aqi_24h
- target_aqi_48h
- target_aqi_72h

These values represent future observations and are used only during supervised model training.

## Forecast Horizons

The system predicts:

- AQI +24 hours
- AQI +48 hours
- AQI +72 hours