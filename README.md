# Awesome-AI4CFD

<img src="./images/main.png" width="96%" height="96%">

This review explores the recent advancements in enhancing Computational Fluid Dynamics (CFD) through Machine Learning (ML). The literature is systematically classified into three primary categories: Data-driven Surrogates, Physics-Informed Surrogates, and ML-assisted Numerical Solutions. Subsequently, we highlight applications of ML for CFD in critical scientific and engineering disciplines, including aerodynamics, atmospheric science, and biofluid dynamics, among others.

<font size=6><center><b> Awesome-AI4CFD </b> </center></font>

- [Awesome-AI4CFD](#awesome-ai4cfd)
  - [Existing Benchmarks](#existing-benchmarks)
  - [Data-driven Surrogates](#data-driven-surrogates)
    - [Dependent on Discretization](#dependent-on-discretization)
      - [On Structured Grids](#on-structured-grids)
      - [On Unstructured Mesh](#on-unstructured-mesh)
      - [On Lagrangian Particle](#on-lagrangian-particle)
    - [Independent on Discretization](#independent-on-discretization)
      - [Deep Operator Network](#deep-operator-network)
      - [In Physical Space](#in-physical-space)
      - [Fourier Neural Operator](#fourier-neural-operator)
  - [Physics-driven Surrogates](#physics-driven-surrogates)
    - [Physics-Informed Neural Network (PINN)](#physics-informed-neural-network-pinn)
    - [Discretized PDE-Informed Neural Network](#discretized-pde-informed-neural-network)
  - [ML-assisted Numerical Solutions](#ml-assisted-numerical-solutions)
    - [Assist Simulation at Coarser Scales](#assist-simulation-at-coarser-scales)
    - [Preconditioning](#preconditioning)
    - [Miscellaneous](#miscellaneous)
  - [Application Novelty](#application-novelty)
    - [Aerodynamics](#aerodynamics)
    - [Combustion \& Reacting Flow](#combustion--reacting-flow)
    - [Atmosphere \& Ocean Science](#atmosphere--ocean-science)
    - [Biology Fluid](#biology-fluid)
    - [Plasma](#plasma)
  - [Contributing](#contributing)

---


## Existing Benchmarks

|  Title  |   Venue  |   Date   |   Code   |   Note   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/pdebench/PDEBench) <br> [**PDEBench**](https://arxiv.org/abs/2210.07182) <br> | NeurIPS 2022 | 2022-10-13 | [GitHub](https://github.com/pdebench/PDEBench) | Local Demo |
|![Star](https://img.shields.io/github/stars/lululxvi/deepxde.svg?style=social&label=Star) <br> [**DeepXDE: A Deep Learning Library for Solving Differential Equations**](https://epubs.siam.org/doi/pdf/10.1137/19M1274067) <br> | SIAM Review | 2021-01 | [GitHub](https://github.com/lululxvi/deepxde) | - |

---

## Data-driven Surrogates

### Dependent on Discretization

#### On Structured Grids

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/Rose-STL-Lab/Turbulent-Flow-Net.svg?style=social&label=Star) <br> [**Towards Physics-informed Deep Learning for Turbulent Flow Prediction**] <br>  | KDD 2020 | 2020-08-20 | [Github](https://github.com/Rose-STL-Lab/Turbulent-Flow-Net) | - |
|![Star](https://img.shields.io/github/stars/Rose-STL-Lab/Equivariant-Net.svg?style=social&label=Star) <br> [**Incorporating Symmetry into Deep Dynamics Models for Improved Generalization**](https://arxiv.org/pdf/2002.03061.pdf) <br> | ICLR 2021 | 2021-03-15 | [Github](https://github.com/Rose-STL-Lab/Equivariant-Net) |
| <br> [**Approximately Equivariant Networks for Imperfectly Symmetric Dynamics**](https://proceedings.mlr.press/v162/wang22aa/wang22aa.pdf) <br> | ICML 2022 | 2022-06-28 | [Github](https://github.com/Rose-STL-Lab/Approximately-Equivariant-Nets) |

#### On Unstructured Mesh

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<!--GNS--> ![Star](https://img.shields.io/github/stars/google-deepmind/deepmind-research.svg?style=social&label=Star) <br> [**Learning to Simulate Complex Physics with Graph Networks**](https://arxiv.org/abs/2002.09405) <br> | ICML 2020 | 2020-09-14 | [Github](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) | [Video](https://sites.google.com/view/learning-to-simulate) |
|<!--MGN--> ![Star](https://img.shields.io/github/stars/google-deepmind/deepmind-research.svg?style=social&label=Star) <br> [**Learning Mesh-Based Simulation with Graph Networks**](https://arxiv.org/abs/2010.03409) | ICLR 2021 <br>| 2021-06-18 | [Github](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | [Video](https://sites.google.com/view/meshgraphnets) |
|<!--MP-PDE--> ![Star](https://img.shields.io/github/stars/brandstetter-johannes/MP-Neural-PDE-Solvers.svg?style=social&label=Star) <br> [**Message Passing Neural PDE Solvers**](https://arxiv.org/abs/2202.03376)<br> | ICLR 2022 | 2023-03-20 | [Github](https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers) | Local Demo|
|![Star](https://img.shields.io/github/stars/jaggbow/magnet.svg?style=social&label=Star) <br> [**MAgNet:Mesh Agnostic Neural PDE Solver**](https://arxiv.org/pdf/2210.05495.pdf)<br> | NeurIPS 2022 | 2023-03-20 | [Github](https://github.com/jaggbow/magnet) | Local Demo|
|<br>[**CARE:Modeling Interacting Dynamics Under Temporal Environmental Variation**](https://openreview.net/attachment?id=lwg3ohkFRv&name=pdf)<br> | NeurIPS 2023 | 2023-12-15 | - | - |

#### On Lagrangian Particle

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/isl-org/DeepLagrangianFluids.svg?style=social&label=Star)<br>[**LAGRANGIAN FLUID SIMULATION WITH CONTINUOUS CONVOLUTIONS**](https://openreview.net/pdf?id=B1lDoJSYDH)<br> | ICLR2020 | 2019-09-25 | [Github](https://github.com/isl-org/DeepLagrangianFluids) | - |
|![Star](https://img.shields.io/github/stars/BaratiLab/FGN.svg?style=social&label=Star)<br>[**Graph neural network accelerated lagrangian fluid simulation**](https://dl.acm.org/doi/abs/10.1016/j.cag.2022.02.004)<br> | Comput Graph | 2022-04-01 | [Github](https://github.com/BaratiLab/FGN) | - |
| <br>[**Fast Fluid Simulation via Dynamic Multi-Scale Gridding**](https://ojs.aaai.org/index.php/AAAI/article/view/25255/25027)<br> | AAAI 2023 | 2023-06-26 | - | - |

---

### Independent on Discretization

#### Deep Operator Network

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/placeholder-for-link-to-DeepONet-repo.svg?style=social&label=Star) <br> [**Learning nonlinear operators via deeponet based on the universal approximation theorem of operators.**](https://www.nature.com/articles/s42256-021-00302-5.pdf) | Nature Machine Intelligence | 2021 | [Github](https://github.com/lululxvi/deeponet) |  |
|<br> [**Nomad: Nonlinear manifold decoders for operator learning.**](https://arxiv.org/pdf/2206.03551)<br> | NeurIPS | 2022 | - |  |
| ![Star](https://img.shields.io/github/stars/lululxvi/deepxde.svg?style=social&label=Star) <br> [**Fourier-mionet: Fourier enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration.**](https://arxiv.org/pdf/2303.04778v1.pdf) <br> | J. Comput. Phy. | Date Not Provided | [Github](https://github.com/lululxvi/deepxde) | - |
|<br> [**Hyperdeeponet: learning operator with complex target function space using the limited resources via hypernetwork.**](https://arxiv.org/pdf/2312.15949v1.pdf)<br> |  ICLR | 2023 | - |  |

---

#### In Physical Space

| Title | Venue | Date | Code | Demo |
|:------|:-----:|:----:|:----:|:----:|
| <br> [**Multipole graph neural operator for parametric PDEs**](https://arxiv.org/abs/2006.09535) | NeurIPS  | 2020 | - | - |
| <br> [**Geometry-informed neural operator for large-scale 3d pdes**](https://arxiv.org/pdf/2309.00583) | NeurIPS  | 2023 | - | - |
| ![Star](https://img.shields.io/github/stars/samholt/NeuralLaplace.svg?style=social&label=Star) <br> [**LNO: Laplace neural operator for solving differential equations.**](https://arxiv.org/pdf/2303.10528) | Arxiv  | 2023 | [Github](https://github.com/samholt/NeuralLaplace) | [Video](https://www.youtube.com/watch?v=o799FwH85cw) |
| ![Star](https://img.shields.io/github/stars/Koopman-Laboratory/KoopmanLab.svg?style=social&label=Star) <br> [**Koopman Neural Operator as a mesh-free solver of non-linear PDEs.**](https://arxiv.org/abs/2301.10022) | Arxiv  | 2023 | [Github](https://github.com/Koopman-Laboratory/KoopmanLab) | - |
| ![Star](https://img.shields.io/github/stars/liuyangmage/in-context-operator-networks.svg?style=social&label=Star) <br> [**In-context operator learning for differential equation problems.**](https://placeholder-for-link-to-KNO-paper)<br> | PNAS | 2023 | [Github](https://github.com/liuyangmage/in-context-operator-networks)| - |

---

#### Fourier Neural Operator

| Title | Venue | Date | Code | Demo |
|:------|:-----:|:----:|:----:|:----:|
| ![Star](https://img.shields.io/github/stars/khassibi/fourier-neural-operator.svg?style=social&label=Star) <br> [**Fourier neural operator for parametric partial differential equations.**](https://arxiv.org/pdf/2010.08895) | ICLR  | 2021 | [Github](https://github.com/khassibi/fourier-neural-operator) | [Video](https://zongyi-li.github.io/blog/2020/fourier-pde/) |
| ![Star](https://img.shields.io/github/stars/alasdairtran/fourierflow.svg?style=social&label=Star) <br> [**Factorized fourier neural operators**](https://arxiv.org/abs/2111.13802) | ICLR  | 2023 | [Github](https://github.com/alasdairtran/fourierflow) | - |
| ![Star](https://img.shields.io/github/stars/microsoft/cliffordlayers) <br> [**Clifford neural layers for pde modeling**](https://arxiv.org/pdf/2209.04934) | ICLR  | 2023 | [Github](https://github.com/microsoft/cliffordlayers/tree/main) | - |
| <br> [**Geometry-informed neural operator for large-scale 3d pdes.**](https://arxiv.org/abs/2309.00583) | NeurIPS  | 2023 | -| - |

---

## Physics-driven Surrogates

### Physics-Informed Neural Network (PINN)

| Title | Venue | Date | Code | Demo |
|:------|:-----:|:----:|:----:|:----:|
| ![Star](https://img.shields.io/github/stars/maziarraissi/PINNs.svg?style=social&label=Star) <br> [**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs**](https://placeholder-for-link-to-PINN-paper) | JCP(Journal of Computational Physics) | 2019 | [Github](https://github.com/maziarraissi/PINNs) | - |
| ![Star](https://img.shields.io/github/stars/ZhaoChenCivilSciML/EQDiscovery-1.svg?style=social&label=Star) <br> [**Physics-informed learning of governing equations from scarce data.**](https://arxiv.org/abs/2005.03448) | Nature Communications | 2021 | [Github](https://github.com/ZhaoChenCivilSciML/EQDiscovery-1) | - |
| ![Star](https://img.shields.io/github/stars/Scien42/NSFnet.svg?style=social&label=Star) <br> [**NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations**](https://arxiv.org/abs/2003.06496) <br> | JCP(Journal of Computational Physics) | 2021 | [Github](https://github.com/Scien42/NSFnet) | - |
| <br> [**Meta-auto-decoder for solving parametric partial differential equations.**](https://arxiv.org/pdf/2111.08823) <br> | NeurIPS | 2022 |- | - |
|<br>  [**Nas-pinn: neural architecture search-guided physics-informed neural network for solving pdes.**](https://arxiv.org/pdf/2305.10127) <br> | JCP(Journal of Computational Physics) | 2024 | - | - | 

---

### Discretized PDE-Informed Neural Network

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/YDuME/EDNN.svg?style=social&label=Star) <br> [**Evolutional Deep Neural Network**](https://arxiv.org/pdf/2103.09959.pdf) <br>| Physical Review E | 2021-10-04 | [Github](https://github.com/YDuME/EDNN) | - |
[**Adlgm: An efficient adaptive sampling dl galerkin method**](https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999123X00035/1-s2.0-S0021999123000396/main.pdf) <br>| JCP |  | - | - |
| ![Star](https://img.shields.io/github/stars/honglin-c/INSR-PDE) <br> [**Implicit Neural Spatial Representations for Time-dependent PDEs**](https://arxiv.org/pdf/2210.00124.pdf) <br>| ICML 2023 | 2022-09-29 | [Github](https://github.com/honglin-c/INSR-PDE) | - |
| ![Star](https://img.shields.io/github/stars/pehersto/ng.svg?style=social&label=Star) <br> [**Neural galerkin schemes with active learning for high dimensional evolution equations**](https://arxiv.org/pdf/2203.01360.pdf) <br>| JCP | 2024-01 | [Github](https://github.com/pehersto/ng) | - |
[**Multi-resolution partial differential equations preserved learning framework for spatiotemporal dynamics**](https://arxiv.org/abs/2205.03990) <br>| Communications Physics | 2024-01-13 | [Github](https://github.com/bitzhangcy/Neural-PDE-Solver) | - |

---

## ML-assisted Numerical Solutions

### Assist Simulation at Coarser Scales

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/google/data-driven-discretization-1d) <br> [**Learning data-driven discretizations for partial differential equations.**](https://arxiv.org/abs/1808.04930) <br>| PNAS | 2019 | [Github](https://github.com/google/data-driven-discretization-1d) | - |
|  <br> [**Machine learning accelerated computational fluid dynamics.**](https://arxiv.org/abs/2102.01010) <br>| PNAS | 2021 | - | - |
| ![Star](https://img.shields.io/github/stars/tum-pbs/differentiable-piso.svg?style=social&label=Star) <br> [**Learned turbulence modelling with differentiable fluid solvers: physics-based loss functions and optimisation horizons.**](https://arxiv.org/abs/2202.06988) <br>| JFM(Journal of Fluid Mechanics) | 2022 | [Github](https://github.com/tum-pbs/differentiable-piso) | - |
| <br> [**Machine learning design of volume of fluid schemes for compressible flows.**](https://www.sciencedirect.com/science/article/abs/pii/S0021999120300498) <br>| JCP(Journal of Computational Physics) | 2020 | - | - |
| ![Star](https://img.shields.io/github/stars/tianhanz/DNN-Models-for-Chemical-Kinetics.svg?style=social&label=Star) <br> [**A multi-scale sampling method for accurate and robust deep neural network to predict combustion chemical kinetics**](https://arxiv.org/abs/2201.03549) <br>|  | 2022 | [Github](https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics) | - |
| ![Star](https://img.shields.io/github/stars/Edward-Sun/TSM-PDE.svg?style=social&label=Star) <br> [**A neural pde solver with temporal stencil modeling.**](https://arxiv.org/abs/2302.08105) <br>| arXiv. | 2023 | [Github](https://github.com/Edward-Sun/TSM-PDE) | - |
| <br> [**Scalable projection-based RO models for large multiscale fluid systems.**](https://ui.adsabs.harvard.edu/link_gateway/2023AIAAJ..61.4499W/PUB_HTML) <br>| AIAA | 2023 | - | - |

---

### Preconditioning

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/danielgreenfeld3/Learning-to-optimize-multigrid-solvers) <br> [**Learning to simulate complex physics with graph networks**](https://arxiv.org/pdf/2002.09405) <br>| ICML | 2020 | [Github](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) | [video](sites.google.com/view/learning-to-simulate) |
| ![Star](https://img.shields.io/github/stars/danielgreenfeld3/Learning-to-optimize-multigrid-solvers) <br> [**Learning to optimize multigrid pde solvers**](https://arxiv.org/pdf/1902.10248) <br>| ICML | 2019 | [Github](https://github.com/danielgreenfeld3/Learning-to-optimize-multigrid-solvers) | - |
| ![Star](https://img.shields.io/github/stars/ilayluz/learning-amg.svg?style=social&label=Star) <br> [**Learning algebraic multigrid using graph neural networks.**](https://arxiv.org/pdf/2003.05744) <br>| ICML | 2020 | [Github](https://github.com/ilayluz/learning-amg) | - |

---

### Miscellaneous

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [**Using ML to augment coarse-grid CFD simulations.**](https://arxiv.org/pdf/2010.00072) <br>| arXiv. | 2020 |-| - |
| ![Star](https://img.shields.io/github/stars/tum-pbs/Solver-in-the-Loop.svg?style=social&label=Star) <br> [**Solver-in-the-loop:Learning from differentiable physics to interact with iterative pde-solvers**](https://arxiv.org/abs/2007.00016) <br>|NeuraIPS  | 2020 | [Github](https://github.com/tum-pbs/Solver-in-the-Loop) | - |
| ![Star](https://img.shields.io/github/stars/locuslab/cfd-gcn.svg?style=social&label=Star) <br> [**Combining differentiable pde solvers and graph neural networks for fluid flow prediction**](https://arxiv.org/pdf/2007.04439) <br>| ICML | 2020 | [Github](https://github.com/locuslab/cfd-gcn) | - |
| <br> [**A deep learning based accelerator for fluid simulations**](https://arxiv.org/pdf/2005.04485) <br>| ICS | 2020 | - | - |

---

## Application Novelty

### Aerodynamics

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/aerorobotics/neural-fly.svg?style=social&label=Star)<br> [**Neural-fly enables rapid learning for agile flight in strong winds**](https://arxiv.org/pdf/2205.06908) <br> | Science Robotics | 2022 | [Github](https://github.com/aerorobotics/neural-fly) | - |
|![Star](https://img.shields.io/github/stars/areenraj/transonic-flow-supercritical-airfoil.svg?style=social&label=Star)<br> [**Prediction of transonic flow over supercritical airfoils using geometric encoding and deep-learning strategies**](https://arxiv.org/pdf/2303.03695) <br> | Physics of Fluids |2023 | [Github](https://github.com/areenraj/transonic-flow-supercritical-airfoil) | - |
|<br> [**Shock wave prediction in transonic flow fields using domain-informed probabilistic deep learning.**](https://www.researchgate.net/profile/Bilal-Mufti-3/publication/377320644_Shock_wave_prediction_in_transonic_flow_fields_using_domain-informed_probabilistic_deep_learning/links/65a477e0af617b0d8744e8ea/Shock-wave-prediction-in-transonic-flow-fields-using-domain-informed-probabilistic-deep-learning.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) <br> | Physics of Fluids | 2024 | - | - |

---

### Combustion & Reacting Flow

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|![Star](https://img.shields.io/github/stars/DENG-MIT/Stiff-PINN)<br> [**Stiff-pinn: Physics-informed neural network for stiff chemical kinetics**]([https:/](https://arxiv.org/pdf/2011.04520)) <br> | The Journal of Physical Chemistry A | 2021 | [Github](https://github.com/DENG-MIT/Stiff-PINN)  | - |
|<br> [**A multi-scale sampling method for accurate and robust deep neural network to predict combustion chemical kinetics**](https://arxiv.org/pdf/2201.03549) <br> | Combustion and Flame | 2022 | [Github](https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics) | - |

---

### Atmosphere & Ocean Science

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [**FourCastNet: A global data-driven HR weather model using AFNO**](https://arxiv.org/pdf/2202.11214) <br> | arXiv. | 2022 | - | - |
|![Star](https://img.shields.io/github/stars/198808xc/Pangu-Weather)<br> [**Accurate medium-range global weather forecasting with 3d nns.**](https://www.nature.com/articles/s41586-023-06185-3.pdf) <br> | Nature | 2023 | [Github](https://github.com/198808xc/Pangu-Weather) | - |
|![Star](https://img.shields.io/github/stars/google-deepmind/graphcast)<br>  [**Learning skillful medium-range global weather forecasting**](https://arxiv.org/pdf/2212.12794) <br> | Science | 2023 | [Github](https://github.com/google-deepmind/graphcast) | - |
|<br> [**Evaluation of Deep Neural Operator models toward ocean forecasting**](https://arxiv.org/pdf/2308.11814) <br> | OCEANS | 2023 | - | - |
|![Star](https://img.shields.io/github/stars/gegewen/ufno)<br> [**U-fno an enhanced FNO-based DL model for multiphase flow**](https://arxiv.org/pdf/2109.03697) <br> | AWR | 2022 | [Github](https://github.com/gegewen/ufno) | - |
|<br> [**Fourier-mionet: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration.**](https://arxiv.org/pdf/2303.04778) <br> | arXiv. | 2023 | - | - |

---

### Biology Fluid

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [**Simulating progressive intramural damage leading to aortic dissection using DeepONet: an operator–regression neural network**](https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2021.0670) <br> | Journal of The Royal Society Interface | 2022 | - | - |
|<br> [**Improving microstructural integrity, interstitial fluid, and blood microcirculation images.**](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.29753) <br> |  JSMRM |  2023 | - | - |
|<br> [**Multiple case PINN for biomedical tube flows**](https://arxiv.org/pdf/2309.15294.pdf) <br> |  arXiv. |  2023 | - | - |

---

### Plasma

|  Title  |   Venue  |   Date   |   Code   |   Demo   |
|:--------|:--------:|:--------:|:--------:|:--------:|
|<br> [**Low-temperature plasma simulation based on physics-informed neural networks.**]([https://](https://arxiv.org/pdf/2206.15294)) <br> |  Phys. Fluids |  2022 | - | - |
|<br> [**Fourier neural operator for plasma modelling**]([https://](https://arxiv.org/pdf/2302.06542.pdf)) <br> |  arXiv. |  2023 | - | - |

---

## Contributing

📮📮📮 If you want to add your model in our leaderboards, please feel free to fork and update your work in the table. You can also email **<wang.hx@stu.pku.edu.cn>**. We will response to your request as soon as possible!



