# ASD Multimodal Project

This is the VS Code application package for the verified ASD multimodal
machine-learning project.

## Frozen ML branches

- Genetic
- Behavioral
- Environmental
- Multimodal knowledge-level fusion

## Verified fusion weights

| Modality | Weight |
|---|---:|
| Genetic | 0.319370 |
| Behavioral | 0.400219 |
| Environmental | 0.280411 |

## Modality IDs

| ID | Modality |
|---:|---|
| 0 | Genetic |
| 1 | Behavioral |
| 2 | Environmental |

## Verified ALL-THREE P001-P005 results

| Patient | Probability | Prediction |
|---|---:|---|
| P001 | 0.419700 | NON-ASD |
| P002 | 0.191837 | NON-ASD |
| P003 | 0.274240 | NON-ASD |
| P004 | 0.183460 | NON-ASD |
| P005 | 0.561214 | ASD |

## Development rule

Do not retrain or modify the frozen checkpoints.

The application layer loads the existing artifacts and exposes
their inference functionality through the backend API.

## Development order

1. Backend model loading
2. Genetic inference API
3. Behavioral inference API
4. Environmental inference API
5. Exact fusion API
6. ANY-ONE / ANY-TWO / ALL-THREE handling
7. Frontend patient input
8. Modality result cards
9. Fusion dashboard
10. End-to-end regression testing