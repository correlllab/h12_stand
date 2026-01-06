# H12 Stand

A simple basic standing environment for the Unitree H12 humanoid robot. To be integrated in the Humanoid_Simulation that uses Docker.

## Installation

```bash
python -m pip install -e source/h12_stand -q
```

## Training

Train the policy using:

```bash
python scripts/rsl_rl/train.py --task Template-H12-Stand-v0 --num_envs 4096 --headless
```

**Options:**
- `--num_envs`: Number of parallel environments (default: 4096)
- `--headless`: Run without GUI
- `--seed`: Random seed for reproducibility

## Playing / Evaluation

Evaluate the trained policy:

```bash
python scripts/rsl_rl/play.py --task Template-H12-Stand-v0 --num_envs 4
```