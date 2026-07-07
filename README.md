# Online Hierarchical MPC–RL Control for Building Heating without Historical Operating Data

This repository provides the source code, evaluation scripts, configuration files, and simulation results for the article:

**Online Hierarchical MPC–RL Control for Building Heating without Historical Operating Data**, accepted for publication in *Energy*.

The proposed method is designed for safe online building heating control under dynamic electricity pricing, with a focus on cold-start deployment without historical operating data.

## Overview

This project implements an online hierarchical MPC–RL control framework for heat-pump-driven building heating systems. The controller combines:

- a high-level rule-learning fusion mechanism for target temperature generation;
- a low-level MPC teacher and RL student policy for heating action generation;
- a risk-gated soft takeover mechanism to improve thermal safety;
- an online physical model adaptation mechanism based on moving horizon estimation.

The experiments are conducted on the `bestest_hydronic_heat_pump` test case from the BOPTEST platform under a highly dynamic electricity price scenario.

## Repository Structure

```text
.
├── main.py
├── boptestGymEnv.py
├── env_wrapper.py
├── algorithmsDDPGT/
├── teacher/
├── testing/
├── results/
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
