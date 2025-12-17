# Soft-Square Inverse Physics (Taichi + Gen)

Minimal, fully controllable soft-body “soft square” world (mass–spring system) for studying soft-material inference from motion.  
Project for **CGSC 2740: Algorithms of the Mind (Yale)** — Shawn Nordstrom, Dec 2025.

This repo implements and compares:
1. **Inverse-physics model**: treats a soft-body simulator as a forward generative model and infers latent material parameters (stiffness `k`, damping `c`) from observed motion summaries.
2. **Feature-based baseline**: predicts `k` and `c` directly from hand-designed motion features (e.g., max compression, settling time).

