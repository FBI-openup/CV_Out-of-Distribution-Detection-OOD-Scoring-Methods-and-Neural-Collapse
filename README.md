# Out-of-Distribution Detection and Neural Collapse Analysis

ResNet-18 trained on CIFAR-100 for OOD detection methods and Neural Collapse phenomena analysis.

## Training Environment

### Hardware
- GPU: NVIDIA GPU (CUDA enabled)
- CPU: 24 cores
- Memory: Sufficient for batch size 128

### Software
- Python: 3.12
- PyTorch: Latest stable version
- torchvision: Latest stable version
- CUDA: Compatible version
- Virtual Environment: `~/Aster-WorkSpace/CV/OOD_venv/`

### Training Infrastructure
- Remote training via SSH
- Session management: tmux
- Version control: Git (GitHub)

## Training Configuration

### Model Architecture
- Base Model: ResNet-18
- Modification: Final layer replaced with `Sequential(Dropout(0.55), Linear(512, 100))`
- Total Parameters: ~11M
- Output Classes: 100 (CIFAR-100)

### Hyperparameters
```
Epochs: 200
Batch Size: 128
Learning Rate: 0.1
Momentum: 0.9
Weight Decay: 0.001
Dropout: 0.55
Scheduler: MultiStepLR (milestones=[60, 120, 160], gamma=0.2)
Optimizer: SGD
Loss Function: CrossEntropyLoss
```

### Data Augmentation
- Training: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
- Testing: Normalize only

### Training Strategy
- Strong regularization (dropout 0.55, weight decay 1e-3)
- Focus: Balance generalization vs accuracy
- Trade-off: Intentionally sacrificed some accuracy for overfitting prevention

## Training Results

### Performance Metrics
```
Train Accuracy: 70.27%
Test Accuracy: 58.99%
Overfitting Gap: 11.28%
```

### Analysis
- Moderate test accuracy for CIFAR-100 (100-class problem)
- Successful overfitting control (gap reduced from 19% to 11%)
- Good generalization achieved through regularization

## Neural Collapse Analysis

This project analyzes Neural Collapse (NC) phenomena in the trained model. Neural Collapse is a geometric phenomenon occurring in the final layers of well-trained neural networks.

### NC Phenomena Overview

#### NC1: Variability Collapse
- Within-class features collapse to class mean centers
- Metric: Within-class variance
- Expected: Variance approaching 0

#### NC2: Simplex ETF (Equiangular Tight Frame)
- Class mean centers form maximally separated geometric structure
- Metric: Pairwise cosine similarity between centered class means
- Expected: Cosine similarity = -1/(C-1) = -0.0101 for 100 classes

#### NC3: Self-Duality
- Class mean features align with classifier weights
- Metric: Correlation between class means and final layer weights
- Expected: Correlation approaching 1

#### NC4: Nearest Class Center
- Model predictions match nearest class center decisions
- Metric: Consistency percentage
- Expected: >95% consistency

#### NC5: ID/OOD Orthogonality
- In-distribution and out-of-distribution data are orthogonal in feature space
- Metric: Spatial separation in t-SNE/feature space
- Expected: Clear separation between ID and OOD

## Visualization Figures

### Training Results
| Figure | Description | Related Question |
|--------|-------------|-----------------|
| `training_curves.png` | Loss and accuracy curves over 200 epochs | Q1: Model Training |

### Neural Collapse Visualizations

#### NC1: Variability Collapse
| Figure | Description | Analysis |
|--------|-------------|----------|
| `NC1_variance.png` | Within-class variance bar chart and distribution | Partial NC1: Mean variance 0.100 |
| `NC1_tsne.png` | t-SNE 2D visualization of feature clustering | Shows moderate class clustering |

**Result**: Partial NC1 - Features show moderate collapse with mean variance 0.10 (expected <0.01 for strong NC1)

#### NC2: Simplex ETF Structure
| Figure | Description | Analysis |
|--------|-------------|----------|
| `NC2_etf_structure_not_centered.png` | INCORRECT: Non-centered cosine similarity analysis | Wrong result: Mean 0.54 (positive) |
| `NC2_etf_structure.png` | CORRECTED: Centered cosine similarity, angle distribution, heatmap, deviation | Critical fix: Added global mean centering |

**Important Note**: Initial NC2 implementation missing centering step. Corrected version properly centers class means relative to global mean before computing ETF structure.

**Result**: After correction, proper ETF measurement relative to global center

#### NC3: Self-Duality
| Figure | Description | Analysis |
|--------|-------------|----------|
| `NC3_self_duality.png` | Scatter plot, correlation distribution, angle distribution, cross-correlation heatmap | Partial+ NC3: Mean correlation 0.887 |

**Result**: Partial+ NC3 - Strong alignment between class means and classifier weights (correlation 0.887, all classes >0.8)

#### NC4: Nearest Class Center
| Figure | Description | Analysis |
|--------|-------------|----------|
| `NC4_nearest_center.png` | Consistency per class, distribution, mismatch heatmap, sample count analysis | Strong- NC4: 91.43% consistency |

**Result**: Strong- NC4 - Model predictions closely match nearest-center decisions (91.43%, approaching 95% threshold)

#### NC5: ID/OOD Orthogonality
| Figure | Description | Status |
|--------|-------------|--------|
| To be implemented | t-SNE visualization, distance distributions, orthogonality metrics | Pending |

## Key Findings

### Neural Collapse Summary

| Phenomenon | Metric | Result | Strength | Interpretation |
|------------|--------|--------|----------|----------------|
| NC1 | Variance | 0.100 | Partial | Moderate within-class collapse |
| NC2 | Cos Sim | Centered | Corrected | Proper ETF measurement after fixing centering |
| NC3 | Correlation | 0.887 | Partial+ | Good mean-weight alignment |
| NC4 | Consistency | 91.43% | Strong- | Near-perfect nearest-center classification |
| NC5 | Separation | TBD | Pending | ID/OOD analysis pending |

### Insights

1. **Regularization Impact**: Strong regularization (dropout 0.55) prevents perfect NC while maintaining good generalization

2. **Partial NC Pattern**: Model exhibits partial Neural Collapse - intermediate state between random initialization and perfect NC

3. **NC2 Correction**: Critical learning - ETF structure must be measured relative to global mean center, not absolute directions

4. **NC Independence**: NC3 and NC4 can be strong even when NC1-NC2 are partial, showing phenomena can emerge independently

5. **Practical Trade-off**: Results demonstrate realistic training scenario prioritizing generalization over perfect geometric structure

## File Structure

```
OOD-Analysis/
├── README.md                              # This file
├── Resnet_OOD.ipynb                      # Main implementation notebook
├── Resnet_OOD.py                         # Script version for tmux training
├── tmux_commands.md                      # Remote training guide
├── resnet18_cifar100.pth                 # Trained model weights
├── training_curves.png                   # Training visualization
├── NC1_variance.png                      # NC1 analysis
├── NC1_tsne.png                          # NC1 t-SNE visualization
├── NC2_etf_structure_not_centered.png   # NC2 incorrect (kept for learning)
├── NC2_etf_structure.png                # NC2 corrected
├── NC3_self_duality.png                 # NC3 analysis
├── NC4_nearest_center.png               # NC4 analysis
└── data/                                 # CIFAR-100 dataset (not tracked)
```

## Usage

### Training
```bash
# Activate virtual environment
source ~/Aster-WorkSpace/CV/OOD_venv/bin/activate

# Run training (200 epochs, ~12-15 hours on GPU)
python Resnet_OOD.py

# Or use Jupyter notebook
jupyter notebook Resnet_OOD.ipynb
```

### Remote Training with tmux
See `tmux_commands.md` for detailed instructions on remote training setup and session management.

### Neural Collapse Analysis
Run the corresponding cells in `Resnet_OOD.ipynb`:
- NC1: Cells for feature extraction and variance analysis
- NC2: Cells for centered ETF structure analysis (corrected version)
- NC3: Cells for self-duality verification
- NC4: Cells for nearest-center consistency check
- NC5: To be implemented

## Future Work

1. Complete NC5 (ID/OOD Orthogonality) analysis
2. Implement bonus task: NC analysis on earlier layers (layer1, layer2, layer3)
3. Compare OOD detection methods: MSP, Max Logit, Energy Score, Mahalanobis, ViM, NECO
4. Explore training with reduced regularization for stronger NC phenomena
5. Analyze trade-off between NC strength and generalization quantitatively

## References

- Neural Collapse literature (papers documenting NC1-NC5 phenomena)
- ResNet architecture (He et al.)
- CIFAR-100 dataset
- OOD detection methods

## Authors

Boyuan Zhang
Ecole Polytechnique
2024

## License

Academic use only
