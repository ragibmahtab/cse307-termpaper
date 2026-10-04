# CSE-307 Term Paper - Track 1: Learned Page Replacement

FIFO, LRU and Optimal page replacement compared with a small decision-tree based
eviction policy, on a trace where the access pattern changes halfway.

## Setup
pip install numpy pandas scikit-learn matplotlib

## How to run
cd src
python test_paging.py
python run_experiments.py 20

The second command runs 20 random seeds (about a minute) and saves the tables and
charts in the results folder.

## Folders
- src: paging.py (algorithms + learned policy), test_paging.py, run_experiments.py
- results: raw_results.csv, summary.csv, degradation_table.csv, faults_table.csv, two figures
- report: report.tex and report.pdf

## Setup of the experiment
20000 references, 200 pages, 20 frames. First 10000 references have locality (small
drifting working set plus some sequential scan), the last 10000 are random with short bursts.

## Summary of results (mean of 20 seeds)
| Policy | Hit ratio before | Hit ratio after |
|---|---|---|
| FIFO | 0.814 | 0.371 |
| LRU | 0.883 | 0.372 |
| Learned (static) | 0.889 | 0.341 |
| Learned (online) | 0.888 | 0.355 |
| OPT | 0.905 | 0.561 |

The learned policy is about the same as LRU before the shift and worse after it.

## AI assistance
I used an AI assistant (Claude) to help write the simulator and experiment code.
I ran the experiments and wrote the analysis myself.
