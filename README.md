# SolarWind_Prediction

Predicting **solar wind MHD parameters at 1 AU** with **ConvLSTM** (convolutional-recurrent) deep learning models trained on **heliospheric MHD simulation data** from [Predictive Science Inc. (PSI)](https://www.predsci.com/).

The Sun continuously emits a magnetized plasma — the *solar wind* — that fills the heliosphere and drives space weather at Earth (1 AU). Physics-based magnetohydrodynamic (MHD) models such as PSI's reproduce this flow but are computationally expensive. This project asks a complementary question: **can a spatiotemporal neural network learn the evolution of the solar wind directly from MHD snapshots and forecast future states?** Each Carrington-rotation run is treated as a sequence of 2-D field maps, and a stacked `ConvLSTM2D` encoder–decoder is trained to predict the next frames — i.e. the radial magnetic field and radial velocity as the wind propagates outward toward 1 AU.

> **Status / provenance:** Research/experimental code from 2020 (commit history preserved). Mirrored from a private repository. Originally written against TensorFlow/Keras 1.x–2.x (`keras.layers.convolutional_recurrent.ConvLSTM2D`).

---

## The data

The input is **MHD model output from PSI's public heliospheric runs**, downloaded per **Carrington Rotation (CR)** from:

```
http://www.predsci.com/data/runs/cr<NNNN>-medium/<model>/helio/
```

For each available run, four HDF (`.hdf`) volumes are retrieved:

| File          | Quantity                                   |
|---------------|--------------------------------------------|
| `br_r0.hdf`   | Radial magnetic field **Br** (inner boundary) |
| `vr_r0.hdf`   | Radial solar wind velocity **Vr** (inner boundary) |
| `br002.hdf`   | Radial magnetic field **Br** (heliospheric grid) |
| `vr002.hdf`   | Radial solar wind velocity **Vr** (heliospheric grid) |

`ExtractData.py` / `ExtractData.ipynb` enumerate the available CR runs, probe each URL, download the HDF files into `br_file/`, `vr_file/`, `br002_file/`, `vr002_file/`, and index them by rotation. Each volume is read into a 2-D spatial grid (≈ `128 × 110`) and stacked over rotations to form the time sequences fed to the models.

> Data is **not** committed to this repo — run `ExtractData.py` to fetch it directly from PSI.

---

## The models

All models are `ConvLSTM2D` **encoder–decoder** networks (Keras) that take a sequence of 2-D field maps with input shape `(timesteps, 128, 110, 1)` and predict subsequent frames. Training uses **MSE** loss with the **Adam** optimizer (MAE reported as a metric), L2 kernel regularization, and `BatchNormalization` between recurrent-convolutional layers.

| Notebook | Description |
|----------|-------------|
| `Notebooks/2-LayerConvLstm.ipynb` | 2-layer stacked ConvLSTM encoder–decoder (baseline). |
| `Notebooks/3-LayerConvLSTM.ipynb` | 3-layer stacked ConvLSTM encoder–decoder (deeper variant). |
| `Notebooks/2048-Nodes-2LayerConvLSTM.ipynb` | Higher-capacity 2-layer variant (wide hidden state). |
| `Notebooks/2-Layer-PredictionModel.ipynb` | Inference / multi-step forecasting with the 2-layer model. |
| `Notebooks/3-Layer-PredictionModel.ipynb` | Inference / multi-step forecasting with the 3-layer model. |
| `Notebooks/DataAugmentation.ipynb` | Augmentation of the MHD frame sequences to expand training data. |
| `Notebooks/correlation_check.ipynb` | Correlation-based frame de-duplication / quality filtering of the extracted snapshots. |

The encoder ingests an observed window of MHD frames into a latent state; the decoder rolls that state forward to generate predicted Br/Vr maps, which are compared against the held-out MHD frames to evaluate forecast skill.

---

## Repository layout

```
SolarWind_Prediction/
├── ExtractData.py            # Download + index PSI heliospheric MHD runs (Br/Vr HDF files)
├── ExtractData.ipynb         # Notebook version of the data-extraction pipeline
└── Notebooks/
    ├── 2-LayerConvLstm.ipynb
    ├── 3-LayerConvLSTM.ipynb
    ├── 2048-Nodes-2LayerConvLSTM.ipynb
    ├── 2-Layer-PredictionModel.ipynb
    ├── 3-Layer-PredictionModel.ipynb
    ├── DataAugmentation.ipynb
    └── correlation_check.ipynb
```

---

## Getting started

```bash
# 1. Dependencies (research code; pin versions to taste)
pip install wget numpy h5py matplotlib scikit-learn tensorflow keras

# 2. Download the PSI heliospheric MHD data
python ExtractData.py
#    -> populates br_file/ vr_file/ br002_file/ vr002_file/

# 3. Explore / train
jupyter notebook Notebooks/
#    Start with correlation_check.ipynb (data QA), then a ConvLSTM training notebook.
```

Reading the `.hdf` volumes requires an HDF reader (`h5py`, or PSI's own tools). A GPU is recommended for training the ConvLSTM models.

---

## Pipeline at a glance

```
PSI heliospheric MHD runs (per Carrington Rotation)
        │   br_r0 / vr_r0 / br002 / vr002  (.hdf)
        ▼
ExtractData.py ──► 2-D Br/Vr field maps, indexed by rotation
        │
        ▼
correlation_check + DataAugmentation ──► cleaned frame sequences
        │
        ▼
ConvLSTM2D encoder–decoder (2-layer / 3-layer / 2048-node)
        │
        ▼
Predicted solar-wind MHD parameters (Br, Vr) → 1 AU
```

---

## Acknowledgements

- Heliospheric MHD model data courtesy of **[Predictive Science Inc. (PSI)](https://www.predsci.com/)** and their publicly available simulation runs.
- Built with **TensorFlow / Keras** (`ConvLSTM2D`).

## License

No license file is currently included. If you intend this to be reusable, consider adding one (e.g. MIT). Until then, all rights reserved by the author.
