# Reading Log — Student A

> Deliverable for Step 1. Goal: general understanding, not a deep dive.
> For each paper note (a) the one-sentence takeaway and (b) how it informs *our* method.

## Must-read

### Al-Hourani, Kandeepan, Lardner (2014) — "Optimal LAP Altitude for Maximum Coverage", IEEE WCL
- **Takeaway:** _TODO_ — LoS probability is a sigmoid in elevation angle; optimal UAV altitude trades off path loss vs LoS probability.
- **How it informs us:** This is the channel model already implemented in the simulator (`los_probability`, `air_to_ground_path_loss_dB`). The surrogate learns to approximate its geometry-aware capacity output. Environment params `a, b, eta_los, eta_nlos, sigma_dB` live in `AG_ENV`.

### Mozaffari, Saad, Bennis et al. (2019) — "A Tutorial on UAVs for Wireless Networks", IEEE COMST
- **Takeaway:** _TODO_
- **How it informs us:** _TODO — placement / coverage framing, fairness metrics._

### Milojevic-Dupont et al. (2023) — "EUBUCCO v0.1", Nature Scientific Data
- **Takeaway:** _TODO_ — ~206M EU buildings, ~74% with height, ODbL license.
- **How it informs us:** Primary building-footprint + height source for our 20 cities. Format: GeoPackage per country/region. Citation required in paper.

## ML methodology

### Kendall & Gal (2017) — "What Uncertainties Do We Need in Bayesian Deep Learning?", NeurIPS
- **Takeaway:** _TODO_ — aleatoric (data) vs epistemic (model) uncertainty; heteroscedastic regression predicts input-dependent variance.
- **How it informs us:** Justifies our two-head MLP (mean + log-variance) and the Gaussian NLL loss. Enables the lower-confidence-bound optimizer (Step 11 stretch goal).

### Goodfellow, Bengio, Courville — Deep Learning, Ch. 6–8
- **Takeaway:** _TODO_ (only if PyTorch/MLPs are new).

## Related work to cite

### Bakirtzis et al. (2022) — "Deep learning for 5G/6G mmWave channel modeling"
- **Note:** _TODO_

### Hoydis et al. (2022) — "Sionna: open-source library for physical-layer research"
- **Note:** _TODO_

### Levie et al. (2021) — "RadioUNet: Fast radio map estimation with CNNs"
- **Note:** _TODO — closest prior art (learned radio maps); contrast: we use a tiny feature-based MLP, not a CNN over maps._

## "Learned channel models" (2–3 papers, Student A to find)
- _TODO_
- _TODO_
