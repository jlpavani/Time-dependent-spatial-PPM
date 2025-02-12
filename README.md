# Time-dependent-spatial-PPM

Data and R code to support Pavani and Quintana (2025).  In this paper, we develop a flexible Bayesian multivariate spatio-temporal model where temporal dependence is defined for areal clusters. The model features a prior distribution for the random partition of areal data that incorporates neighboring information. It also incorporates an autoregressive structure and terms related to seasonal patterns into temporal components that are disease- and cluster-specific. Furthermore, it considers a multivariate directed acyclic graph autoregressive structure to accommodate spatial and inter-disease dependence.We explore the properties of the model through simulation studies and show results that prove our proposal compares well to competing alternatives. Finally, we apply the model to the motivating dataset with a twofold goal: finding clusters of areas with similar temporal trends for some of the diseases and exploring the existence of correlation between two diseases transmitted by the same mosquito.

## Citation

If you find this code helpful and use it in your work, please cite our paper:

> **Pavani, J.**; Quintana, F. A.: A Bayesian multivariate model with temporal dependence on random partition of areal data. *Statistics in Medicine*, 44(3-4), e10325, 2025. [[DOI](https://doi.org/10.1002/sim.10325)]

```bibtex
@Article{Pavani2025,
  author  = {Pavani, Jessica and Quintana, Fernando Andrés},
  journal = {Statistics in Medicine},
  title   = {A {B}ayesian multivariate model with temporal dependence on random partition of areal data for mosquito-borne diseases},
  year    = {2025},
  number  = {3-4},
  pages   = {e10325},
  volume  = {44},
  doi     = {https://doi.org/10.1002/sim.10325},
  eprint  = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.10325},
  url     = {https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.10325},
}
