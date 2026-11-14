# Opposite Hopf Criticalities at a Common Force Limit in an Oscillator--Actuator Model

Reproducibility companion for a manuscript investigating local and nonlocal
bifurcations of an oscillator--actuator system. Contains the original computational
Python scripts and numerical results supplied by the authors. The repository
includes simulation and proof-audit calculations, but these are **not** formal,
computer-assisted proofs and **not** experimental actuator measurements.

## Environment

Python 3 with the package versions listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

Run all commands from the directory containing this README and the `.py` files.
The scripts write/overwrite output CSV, JSON, and PNG files in this directory.

## Main reproduction sequence

```bash
# Figures 2--3: print shooting results; scripts do not save the plots directly
python numerical_validation.py

# Figures 4--5: fixed-parameter shooting, continuation and exact identities
python periodic_continuation.py --outdir .

# Figures 6--7: requires continuation_results.csv from the previous command
python step3_analysis.py --outdir .

# Initial Hopf-criticality data for Figure 8
python saturated_hopf_criticality.py

# Final two-panel Figure 8: depends on step3_analysis.py and
# saturated_hopf_criticality.csv. Overwrites saturated_hopf_criticality.png.
python generalized_hopf_audit.py

# Figure 9: Lyapunov-certificate numerical benchmark
python weak_force_stability.py
```

`generalized_hopf_audit.py` imports computational functions from `step3_analysis.py`;
**do not omit the latter** from a repository that includes the former.
The Figure 8 output file has the same name at both steps; run
`generalized_hopf_audit.py` last to obtain the two-panel figure shown in the paper.

## Supporting numerical and symbolic checks

```bash
python coalescence_verification.py
python independent_checks_step6.py
python independent_checks_step7.py
python independent_hopf_quadrature.py
python neutral_zero_hopf_validation.py
python saturated_hopf_symbolic_audit.py
python independent_center_manifold_audit.py
python independent_fourier_collocation.py
python independent_force_budget_audit.py
python escape_audit.py
```

The `independent_force_budget_audit.py` script reads
`saturated_hopf_criticality.csv` and `generalized_hopf_shooting.csv`,
so execute the Figure 8 scripts first.

## Figure-to-code index

| Manuscript figure | Included image | Main source script | Status |
| --- | --- | --- | --- |
| 1: equilibrium stability atlas | `stability_atlas.png` | Not included | Image only; separate plotting script needed |
| 2: zero--Hopf validation | `validation_ZH.png` | `numerical_validation.py` | Underlying computations reproducible; plotting code not included |
| 3: ordinary Hopf validation | `validation_Hopf.png` | `numerical_validation.py` | Underlying computations reproducible; plotting code not included |
| 4: Hopf loci with cycles | `continuation_atlas.png` | `periodic_continuation.py` | Reproducible |
| 5: amplitudes and period bound | `continuation_branches.png` | `periodic_continuation.py` | Reproducible |
| 6: pseudo-arclength branches | `pseudo_arclength_branches.png` | `step3_analysis.py` | Reproducible |
| 7: saturation sensitivity | `saturation_robustness.png` | `step3_analysis.py` | Reproducible |
| 8: saturated Hopf and Bautin fold | `saturated_hopf_criticality.png` | `saturated_hopf_criticality.py`, `generalized_hopf_audit.py` | Reproducible, sequential execution |
| 9: exponential decay certificate | `weak_force_decay.png` | `weak_force_stability.py` | Reproducible |
| 10: local boundary phase portraits | `local.png` | Not included | Image only; separate plotting script needed |
| 11: global boundary portrait | `global.png` | Not included | Image only; separate plotting script needed |

`validation_results.csv` was supplied, but the provided `numerical_validation.py`
prints its results instead of exporting this CSV or generating the displayed
Figure 2 and 3 PNG files. Their included images and CSV are original supplied assets.
The script that generated `stability_atlas.png` and those generating
`local.png` and `global.png` were not supplied. These are disclosed gaps, not
claims that the underlying mathematics is invalid.

## Data

CSV and JSON files in the repository retain their supplied names. The
`equilibrium_branch_reflection.csv` dataset was additionally regenerated
from the supplied `periodic_continuation.py` during the packaging audit.
No measured physical actuator dataset is included.

## Reproducibility audit (14 November 2026)

All 16 retained Python scripts were executed successfully in an isolated
working copy with the versions listed in `requirements.txt`. The principal
numerical output tables regenerated from the scripts matched the corresponding
supplied CSV data at stored precision. This is a *computational smoke test*;
it does not independently establish every theoretical claim or provide a
certified global bifurcation diagram.

## Citation and licensing

Please cite the associated paper when using these calculations. A permanent
paper citation, DOI and author information should be added once available.
No reuse license has been chosen in this package; the repository owner should
select a license deliberately before inviting reuse.
