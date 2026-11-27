# QThReduction

Numerical code and data accompanying **Finite-Particle Quantum Reduction of Thermodynamic Irreversibility**, by Borhan Ahmadi.

## Versions

- The [original submission snapshot (v1.0-submission)](https://github.com/Borhan19/QThReduction/releases/tag/v1.0-submission) contains the original standalone plotting script and representative cycle audits.
- For the revised manuscript, use the archive `QThReduction-v1.1-revision.zip` supplied with the revision materials. See the [releases page](https://github.com/Borhan19/QThReduction/releases) for versioned downloads.

The original snapshot remains unchanged.

## Revision archive

The revision archive contains the full-precision numerical inputs and output data for the response-matched classical record, paired hard/soft comparisons, strict initial matching, particle-number and duration scans, interaction and larger-lattice tests, common-trajectory control, count-statistics witness, and representative cycle audits.

- `Audit_I/cycle_refined.json`: soft classical cycle control coefficients, work, heat, efficiency and closure residuals.
- `Audit_I/paired_audit.json`: both classical records evaluated on the same thermal trajectories, together with refined thermal calibrations.
- `Audit_II/`: extended numerical and analytical checks.
- `Figs/`: all seven final publication vector figures, including the updated panels 3(a) and 4(a).
- `plot_revision.py` and `plot_additions.py`: regenerate the revised figure set from the saved numerical data.
- `SHA256SUMS.txt`: checksums of the numerical JSON files and archived publication figures.

Extract the revision archive and open a terminal in the extracted directory:

```bash
python -m pip install -r requirements.txt
python Audit_I/verify_audit.py
python Audit_II/verify_stage2.py
python verify_preserved.py
python plot_revision.py
python plot_additions.py
```

The regenerated plots are written to `regenerated_figures/`, without modifying the archived publication PDFs in `Figs/`. The READMEs in the audit directories give commands for rerunning the additional calculations.

## Original standalone script

The original script remains available in the repository as `qthermo_companion_final_plotting_STANDALONE.py`.

To run the original script from the repository directory:

```bash
python -m pip install -r requirements.txt
python qthermo_companion_final_plotting_STANDALONE.py
```

The script generates the original six figures from the finalized numerical inputs and performs an independent representative quantum cycle audit. An optional fresh classical cycle audit can be enabled in the script.

## Reproducibility

Figure regeneration uses stored numerical inputs; it is not a fresh rerun of every original production scan. The additional revision simulations have their own reproduction scripts and archived full-precision outputs. Between-scramble standard errors are sampling uncertainties, not certified total numerical error bounds. Font availability and library versions may affect the appearance of regenerated plots.

## Citation and license

Please cite the accompanying manuscript, **Borhan Ahmadi, Finite-Particle Quantum Reduction of Thermodynamic Irreversibility (2026)**. See `CITATION.cff` and `LICENSE`.
