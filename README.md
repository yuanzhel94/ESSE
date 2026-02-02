# ESSE — Effective Sample Size Estimation

**ESSE (effective sample size estimation)** presents an analytical approach that infers the statistical significance of spatial association tests between spatial maps by estimating the effective sample size of the data.

---

## Installation

We provide both MATLAB and Python implementations of ESSE.

---

### MATLAB

Download the `matlab_esse` folder from the present repository and add it to your MATLAB path:

```matlab
addpath(genpath('matlab_esse'))
```

---

### Python

The Python implementation is distributed via PyPI.  
You may also clone and install manually from the present repository.

> **Note**  
> The package is distributed on PyPI under the name **`pyesse`**,  
> but imported in Python as **`esse`** after installation.

#### PyPI installation

```bash
pip install pyesse
```

#### Import in Python

```python
import esse
```

---

## Example usage

To evaluate the association between spatial maps *x* and *y* given:

- spatial map ***x***, may contain NaN or Inf values.
- spatial map ***y***, may contain NaN or Inf values.
- spatial coordinates ***coord*** of observations

### Stationary ESSE (sESSE)

#### MATLAB

```matlab
% take 70–80s to run (Apple Silicon M1 Pro) on fsaverage5 10k cortical map
% pef - significance p-value
% rX - Pearson correlation coefficient
% nef - effective sample size
[pef, rX, nef, run_status, n_parc, p_naive, fc_para1, fc_para2] = effective_sample_size_estimation(x, y, coord);
```

#### Python

```python
# take ~3.5 min to run (Apple Silicon M1 Pro) on fsaverage5 10k cortical map
# pef - significance p-value
# rX - Pearson correlation coefficient
# nef - effective sample size

import esse
pef, rX, nef, run_status, n_parc, p_naive, fc_para1, fc_para2 = esse.effective_sample_size_estimation(x, y, coord)
```

---

### Nonstationary ESSE (nESSE)

#### MATLAB

```matlab
% nESSE with data-driven parcellation
% take 70–80s to run (Apple Silicon M1 Pro) on fsaverage5 10k cortical map
[pef, rX, nef, run_status, n_parc, p_naive, fc_para1, fc_para2] = effective_sample_size_estimation(x, y, coord, 'xparc', 'auto', 'yparc', 'auto');
```

#### Python

```python
# nESSE with data-driven parcellation
# take ~3.5 min to run (Apple Silicon M1 Pro) on fsaverage5 10k cortical map

import esse
pef, rX, nef, run_status, n_parc, p_naive, fc_para1, fc_para2 = esse.effective_sample_size_estimation(x, y, coord,xparc='auto',yparc='auto')
```

---

## Documentation

For full documentation and additional examples, see
[ESSE documentation](https://esse-effective-sample-size-estimation.readthedocs.io/en/latest/api/esse.html#module-esse.esse)
