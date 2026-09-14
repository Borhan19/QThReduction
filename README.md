# QThReduction

Code accompanying the manuscript

**Finite-Particle Quantum Reduction of Thermodynamic Irreversibility**

by Borhan Ahmadi.

## Overview

This repository contains the numerical and plotting code associated with the
quantum–classical comparison presented in the manuscript.

The main standalone script is

`qthermo_companion_final_plotting_STANDALONE.py`

It contains the finalized numerical inputs used to generate the publication
figures, together with independent consistency checks of the thermodynamic
cycle.

The script distinguishes the exact microscopic state, the record-only
maximum-entropy representative, and the energy-and-record maximum-entropy
representative used in the manuscript.

## Requirements

Python 3.10 or later is recommended.

The code requires:

- NumPy
- pandas
- Matplotlib
- SciPy

Install the dependencies with

```bash
pip install -r requirements.txt
