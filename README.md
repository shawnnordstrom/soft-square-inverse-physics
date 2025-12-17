# Soft-Square Inverse Physics (Taichi + Gen)

Minimal, fully controllable soft-body “soft square” world (mass–spring system) for studying soft-material inference from motion.  
Project for **CGSC 2740: Algorithms of the Mind (Yale)** — Shawn Nordstrom, Dec 2025.

This repo implements and compares:
1. **Inverse-physics model**: treats a soft-body simulator as a forward generative model and infers latent material parameters (stiffness `k`, damping `c`) from observed motion summaries.
2. **Feature-based baseline**: predicts `k` and `c` directly from hand-designed motion features (e.g., max compression, settling time).

**NOTE: I uploaded the files in this repository such that they can each be ran individually. In the report, I used a mix of functions from different files with an overhead controller. The code wasn't as smooth as I'd liked so I've left that out for now!
