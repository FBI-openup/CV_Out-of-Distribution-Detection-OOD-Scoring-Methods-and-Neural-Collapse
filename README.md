# Out-of-Distribution Detection and Neural Collapse Analysis(old data from before NC5)

ResNet-18 trained on CIFAR-100 for OOD detection methods and Neural Collapse phenomena analysis.

## Training Environment

### Hardware
- GPU: NVIDIA GPU (CUDA enabled)
- CPU: 24 cores
- Memory: Sufficient for batch size 128

### Software
- Python: 3.12 venv minconda
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

## Results and Analysis

### Training Results

| Figure | Description | Result |
|--------|-------------|--------|
| `training_curves.png` | Loss and accuracy curves over 200 epochs | Train: 70.27%, Test: 58.99%, Gap: 11.28% |

**Analysis**: Moderate test accuracy for CIFAR-100 (100-class problem). Strong regularization (dropout 0.55) successfully controlled overfitting, achieving good generalization.

---

## Neural Collapse Phenomena Analysis

Neural Collapse (NC) is a geometric phenomenon occurring in the final layers of well-trained neural networks. This section analyzes NC1-NC5 phenomena in our trained ResNet-18 model.

### NC1: Variability Collapse

**Theory**: Within-class features collapse to class mean centers
- Metric: Within-class variance
- Expected: Variance approaching 0

**Visualizations**:

**Figure: `NC1_variance.png`** (2 subplots)
- **Subplot 1 - Bar Chart**: Within-class variance for all 100 classes
  - X-axis: Class ID (0-99)
  - Y-axis: Variance value
  - Purpose: Show per-class collapse strength
- **Subplot 2 - Histogram**: Distribution of variance values
  - X-axis: Variance value
  - Y-axis: Number of classes
  - Purpose: Show overall variance concentration

**Figure: `NC1_tsne.png`** (1 plot)
- **t-SNE 2D Scatter Plot**: Feature space visualization
  - X/Y-axis: t-SNE components
  - Colors: Different classes
  - Purpose: Visual verification of class clustering

**Result**: Partial NC1 - Mean variance 0.100 (expected <0.01 for strong NC1)

---

### NC2: Simplex ETF (Equiangular Tight Frame)

**Theory**: Class mean centers form maximally separated geometric structure
- Metric: Pairwise cosine similarity between centered class means
- Expected: Cosine similarity = -1/(C-1) = -0.0101 for 100 classes

**Visualizations**:

**Figure: `NC2_etf_structure_not_centered.png`** (4 subplots - INCORRECT, kept for learning)
- Initial attempt without centering - Shows positive correlations (wrong)

**Figure: `NC2_etf_structure.png`** (4 subplots - CORRECTED)
- **Subplot 1 - Cosine Similarity Histogram**: Pairwise similarities between centered class means
  - X-axis: Cosine similarity
  - Y-axis: Number of class pairs
  - Purpose: Verify ETF value -0.0101
- **Subplot 2 - Angle Distribution**: Geometric angles between class centers
  - X-axis: Angle (degrees)
  - Y-axis: Number of class pairs
  - Purpose: Verify uniform angular separation
- **Subplot 3 - Heatmap**: Similarity matrix for 20 random classes
  - X/Y-axis: Class IDs
  - Colors: Cosine similarity (blue=negative, red=positive)
  - Purpose: Visual verification of ETF structure
- **Subplot 4 - ETF Deviation**: Distance from theoretical ETF
  - X-axis: Deviation value
  - Y-axis: Frequency
  - Purpose: Measure deviation from perfect ETF

**Critical Fix**: Added global mean centering before computing cosine similarities

**Result**: Proper ETF measurement after centering correction

---

### NC3: Self-Duality

**Theory**: Class mean features align with classifier weights
- Metric: Correlation between class means and final layer weights
- Expected: Correlation approaching 1

**Visualizations**:

**Figure: `NC3_self_duality.png`** (4 subplots)
- **Subplot 1 - Scatter Plot**: Dimensional alignment for one sample class
  - X-axis: Class mean values (512 dimensions)
  - Y-axis: Classifier weight values (512 dimensions)
  - Purpose: Show per-dimension correlation
- **Subplot 2 - Correlation Histogram**: Pearson correlation for all 100 classes
  - X-axis: Correlation coefficient
  - Y-axis: Number of classes
  - Purpose: Primary NC3 metric distribution
- **Subplot 3 - Angle Distribution**: Geometric angles between means and weights
  - X-axis: Angle (degrees)
  - Y-axis: Number of classes
  - Purpose: Secondary alignment metric (note: high correlation can coexist with moderate angles due to scaling)
- **Subplot 4 - Cross-Correlation Heatmap**: Mean-weight correlation matrix for 20 classes
  - X-axis: Weight class ID
  - Y-axis: Mean class ID
  - Colors: Correlation strength
  - Purpose: Visual verification of diagonal alignment

**Result**: Partial+ NC3 - Mean correlation 0.887 (all classes >0.8, std 0.011)

**Note**: Angle of 50.92 degrees reflects varying scaling factors across classes, not misalignment. High correlation is the primary NC3 indicator.

---

### NC4: Nearest Class Center

**Theory**: Model predictions match nearest class center decisions
- Metric: Consistency percentage
- Expected: >95% consistency

**Visualizations**:

**Figure: `NC4_nearest_center.png`** (4 subplots)
- **Subplot 1 - Consistency Bar Chart**: Per-class consistency percentage
  - X-axis: Class ID (0-99)
  - Y-axis: Consistency (%)
  - Purpose: Show which classes follow nearest-center principle
- **Subplot 2 - Consistency Histogram**: Distribution of consistency values
  - X-axis: Consistency (%)
  - Y-axis: Number of classes
  - Purpose: Overall NC4 strength assessment
- **Subplot 3 - Mismatch Heatmap**: Confusion-style matrix for 20 classes
  - X-axis: Model prediction
  - Y-axis: True label
  - Colors: Mismatch count
  - Purpose: Identify systematic deviations
- **Subplot 4 - Sample Count vs Consistency**: Scatter plot
  - X-axis: Samples per class
  - Y-axis: Consistency (%)
  - Purpose: Check if sample size affects consistency

**Result**: Strong- NC4 - 91.43% overall consistency (61/100 classes >90%, all >80%)

---

### NC5: ID/OOD Orthogonality

**Theory**: In-distribution and out-of-distribution data are orthogonal in feature space
- Metric: Spatial separation in t-SNE/feature space
- Expected: Clear separation between ID and OOD

**Status**: To be implemented

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

