# Human Connectome Project (HCP) Distance to Criticality (DTC)

hpc_dtc is a computational neuroscience framework for studying spatial variation in distance to criticality (DTC) across the human cortex using large-scale fMRI data. The repository integrates empirical analyses of resting-state metrics (e.g., fALFF, rFCD) with biologically inspired random recurrent network models to infer cortical proximity to dynamical transitions.
Building on the modeling infrastructure developed in critical_slowing_down_in_free_recall, this project extends the framework to Human Connectome Project (HCP) data and related high-resolution datasets. It provides reproducible pipelines for preprocessing, computing spectral and connectivity-based measures, fitting rate-based network models, and generating publication-quality figures.
The central goal is to quantify and map heterogeneous cortical DTC, testing the hypothesis that large-scale functional specialization reflects structured positioning relative to critical dynamical regimes. The repository is designed for clarity, modularity, and reproducibility, facilitating both methodological development and direct comparison between empirical fMRI observations and network simulations.

## Overview

- 📁 `code/` – Analysis and preprocessing scripts
- 📁 `data/` – Raw and processed datasets, with accompanying metadata
- 📁 `examples/` – Demonstrations for getting started
- 📁 `notebooks/` – Exploratory notebooks for model development and hypothesis testing
- 📁 `results/` – Statistical outputs and visualizations used in the publication
- 📁 `docs/` – Supplementary materials and manuscript source files


## How to install
Clone this repository and install dependencies:

```bash
git clone https://github.com/dovi-yellin/hcp_dtc.git
cd critical_slowing_down_in_free_recall
```

We recommend using a virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Getting Started
The `demos/` folder includes self-contained examples on: (a) how to run a short simulation of the random recurrent network while modulating its proximity to criticality, and (b) how to analyze the results.  



## Citation to related Project
```
@article{yellin2025critical,
  title={Adaptive proximity to criticality underlies amplification of ultra-slow fluctuations during free recall},
  authors={Yellin, Dovi and Siegel, Noam and Malach, Rafael and Shriki, Oren},
  journal={biorxiv},
  year={2025},
  doi={10.1101/2023.02.24.529043}
}
```