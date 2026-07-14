# Sensor Sensitivity Model

Code accompanying the manuscript:

**Modelling In-Situ Vibration Sensor Sensitivity in Water Networks**

Author: Rehan Mendis

## Overview

This repository contains Python code for estimating vibration sensor sensitivity in water distribution networks using:

- Hierarchical bootstrap resampling
- Logistic regression
- Firth penalised logistic regression
- Model performance assessment
- Coefficient uncertainty quantification
- Joint coefficient distribution visualisation

The hierarchical bootstrap accounts for the grouped structure of sensor observations collected at individual leak locations.

---

## Repository Contents

├── sensor_sensitivity_model.py

Main modelling and visualisation functions

├── sensor_leak_data.csv

Sample dataset

├── environment.yml

Conda environment specification

├── README.md

Project documentation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/RehanMendis/sensor-sensitivity-model.git

cd sensor-sensitivity-model