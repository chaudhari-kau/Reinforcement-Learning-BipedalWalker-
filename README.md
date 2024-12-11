# Reinforcement-Learning-BipedalWalker-

Here's a detailed README.md for your project:

```markdown
# SAC vs TD3 for BipedalWalker: A Comparative Analysis

A comparative implementation and analysis of Twin Delayed Deep Deterministic Policy Gradient (TD3) and Soft Actor-Critic (SAC) algorithms for BipedalWalker-v3 and BipedalWalkerHardcore-v3 environments.

## Team Members - Northeastern University
- **Kaustubh Chaudhari** - SAC Implementation
- **Vedant Lohakane** - TD3 Implementation

## Project Structure
BipedalWalker_TD3_SAC/
├── models/                    # Saved model checkpoints
│   ├── td3/
│   │   ├── classic/
│   │   └── hardcore/
│   └── sac/
│       ├── classic/
│       └── hardcore/
├── logs/                     # Training logs
│   ├── td3/
│   └── sac/
├── results/                  # Generated plots and analysis
│   ├── plots/
│   │   ├── comparisons/
│   │   ├── td3/
│   │   └── sac/
│   └── gifs/                 # Recorded agent performances
├── archs/
│   └── ff_models.py          # Neural network architectures
├── environment.py            # Environment wrapper
├── noise.py                  # Noise generators for exploration
├── replay_buffer.py          # Experience replay implementation
├── td3_agent.py             # TD3 algorithm implementation
├── sac_agent.py             # SAC algorithm implementation
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── visualizer.py            # Visualization utilities
├── utils.py                 # Helper functions
├── main_script.py           # Main entry point
└── requirements.txt         # Project dependencies

Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

### Training

1. Interactive Mode (Recommended):
```bash
python main_script.py -i
```
This will provide menu-driven options for:
- Algorithm selection (TD3/SAC)
- Environment selection (Classic/Hardcore)
- Training/Evaluation/Visualization

2. Command Line Mode:
```bash
# Train TD3 on classic environment
python main_script.py --flag train --env classic --rl_type td3

# Train SAC on hardcore environment
python main_script.py --flag train --env hardcore --rl_type sac
```

### Evaluation

1. Generate GIF for specific checkpoint:
```bash
python evaluate.py
# Select Option 1 and enter episode number
```

2. Generate training plots:
```bash
python evaluate.py
# Select Option 2
```

### Visualization and Analysis

Generate comprehensive comparison plots:
```bash
python visualizer.py
```

## Key Files Description

- `ff_models.py`: Custom neural architectures with feed-forward encoders
- `td3_agent.py`: TD3 implementation with dual critics and delayed policy updates
- `sac_agent.py`: SAC implementation with entropy maximization
- `environment.py`: Custom wrapper for BipedalWalker environments
- `train.py`: Training loop with logging and checkpointing
- `evaluate.py`: Evaluation scripts and visualization tools

## Training Parameters

### Common Parameters
- Learning Rate: 4e-4
- Replay Buffer Size: 500,000
- Batch Size: 64 (Classic) / 128 (Hardcore)
- Discount Factor (γ): 0.98
- Soft Update Rate (τ): 0.01
