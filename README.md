# ASD Multimodal Project

## Project Status

The machine-learning portion of this project has been trained,
validated, fused and frozen.

The current package is the application/UI development package.

## Modalities

1. Genetic
2. Behavioral
3. Environmental

## Fusion

Verified knowledge-level fusion weights:

- Genetic: 0.319370
- Behavioral: 0.400219
- Environmental: 0.280411

## Verified ALL-THREE Demonstration Results

| Patient | Probability | Prediction |
|---|---:|---|
| P001 | 0.419700 | NON-ASD |
| P002 | 0.191837 | NON-ASD |
| P003 | 0.274240 | NON-ASD |
| P004 | 0.183460 | NON-ASD |
| P005 | 0.561214 | ASD |

## Demo Data

Exact original P001-P005 demo inputs are stored under:

data/demo/

Each patient contains:

- genetic_input.csv
- behavioral_input.csv
- environmental_input.csv

## Important

Do not retrain or modify the frozen ML artifacts during UI development.

The backend will load the existing trained artifacts and expose
local inference APIs.

The frontend will communicate with the backend through localhost.
