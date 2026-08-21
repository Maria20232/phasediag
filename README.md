# PhaseDiag

Post-training diagnostic framework for Behavior Cloning policies, combining offline imitation metrics, fresh-reset closed-loop evaluation, and phase-conditioned state restoration to characterize policy competence beyond aggregate success rates.

Companion code for the paper *"Beyond Success Rate: Phase-Conditioned Diagnosis of Behavior Cloning Policies for Robotic Manipulation."*

## What this does

Standard BC evaluation reports either offline imitation error (how closely predicted actions match expert actions) or closed-loop success rate (did the task complete). Neither reveals *where* along a task a policy's competence breaks down. PhaseDiag restores a frozen, trained policy to semantically defined phases along expert demonstrations (e.g., approach, grasp, post-grasp) and measures whether it can complete the task from each, producing a phase-wise competence profile instead of one aggregate number.

## Setup

```bash
conda create -n vla python=3.11 -y
conda activate vla
pip install "mujoco==3.3.0" robosuite h5py torch numpy matplotlib
pip install git+https://github.com/ARISE-Initiative/robomimic.git
```

Note: MuJoCo must be pinned to 3.3.0. Newer MuJoCo releases removed the legacy `mjData.qM` attribute that robosuite 1.5.2's controller code depends on.

## Repository structure

## Reproducing the paper's results

1. Collect demonstrations via `robosuite.scripts.collect_human_demonstrations` (Lift: 25 episodes, Stack: 40 episodes).
2. Convert to observations: `robomimic`'s `convert_robosuite.py` then `dataset_states_to_obs.py`.
3. Train: `python scripts/train_bc.py --epochs 500 --seed 42` (repeat for seeds 42/43/44, epochs 50/500).
4. Offline metrics: `python scripts/offline_metrics.py`
5. Closed-loop: `python scripts/eval_bc.py --model <path> --episodes 20`
6. Phase-conditioned: see `scripts/phase_recovery.py`'s `main()` for the full per-seed evaluation loop.

## Important implementation note: controller resynchronization

Restoring a MuJoCo simulator state via `set_state_from_flattened` does **not** automatically resynchronize the OSC controller's cached kinematics. Without an explicit `controller.update(force=True)` call after restoration, the first policy action after a state restore is computed relative to a stale pose, producing spurious failures unrelated to the evaluated policy. See `phase_recovery.py`'s `restore_state()` for the fix.

## Citation

```bibtex
@inproceedings{saibaa2026phasediag,
  title={Beyond Success Rate: Phase-Conditioned Diagnosis of Behavior Cloning Policies for Robotic Manipulation},
  author={Saibaa, Maria and Zito, Claudio},
  year={2026}
}
```

## License

MIT

## Included data

This repository includes the processed demonstration datasets and trained policy checkpoints used in the paper:
- `data/demos_lift/low_dim.hdf5` (25 demonstrations, 18,503 state-action pairs)
- `data/demos_lift/bc_50.pt`, `bc_500.pt`, and seeded variants (`bc_50_seed{42,43,44}.pt`, `bc_500_seed{42,43,44}.pt`)
- `data/demos_stack/low_dim.hdf5` (40 demonstrations, 26,601 state-action pairs)
- `data/demos_stack/bc_stack_50.pt`, `bc_stack_500.pt`

Raw MuJoCo simulator states (needed for state-restoration analyses) are included in each `low_dim.hdf5` file under the `states` key.
