# Comparison Tables

## Qihang Original Data (Epoch 5)

| Model | Val Loss | Top-1 IoU | Top-5 IoU |
|-------|----------|-----------|-----------|
| Candidate | 6.736 | 0.0364 | 0.0798 |
| SupportSurface | 5.965 | 0.0739 | 0.1186 |
| HardConstraint | 5.535 | 0.0637 | 0.1009 |

## New Training (Epoch 5)

| Model | Val Loss | Top-1 IoU | Top-5 IoU |
|-------|----------|-----------|-----------|
| Candidate | 6.010 | 0.0420 | 0.0692 |
| SupportSurface | 5.898 | 0.0734 | 0.1148 |
| HardConstraint | 5.529 | 0.0624 | 0.0996 |

## New Training (Epoch 20)

| Model | Val Loss | Top-1 IoU | Top-5 IoU |
|-------|----------|-----------|-----------|
| Candidate | 6.321 | 0.0483 | 0.0749 |
| SupportSurface | 6.065 | 0.0768 | 0.1137 |
| HardConstraint | 5.935 | 0.0669 | 0.0997 |

## Best Val Top-1 IoU (Across All 20 Epochs)

| Model | Best Top-1 IoU |
|-------|----------------|
| Candidate | 0.0498 |
| SupportSurface | **0.0808** |
| HardConstraint | 0.0673 |
