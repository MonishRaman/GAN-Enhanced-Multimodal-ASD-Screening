# ASD Multimodal UI — Application Architecture

## Frozen ML Layer

The following directories contain the verified project artifacts:

- Genetic_Model/
- Behavioral_Model/
- Environmental_Model/
- 05_multimodal_fusion/
- M4_demo_patients/

These artifacts are frozen.

## Backend

backend/
- app/              FastAPI application
- services/         model inference services
- schemas/          request/response schemas
- models/           application-level model wrappers
- utils/            helper utilities
- tests/            backend tests

## Frontend

frontend/
- src/components/   reusable UI components
- src/pages/        application pages
- src/services/     API communication
- src/types/        TypeScript types
- src/utils/        frontend utilities
- public/           static assets

## Data

data/
- demo/
- uploads/
- results/

## Documentation

docs/

## Intended inference flow

Frontend
    |
    v
FastAPI Backend
    |
    +--> Genetic Service
    |
    +--> Behavioral Service
    |
    +--> Environmental Service
    |
    +--> Fusion Service
              |
              v
       Final ASD probability