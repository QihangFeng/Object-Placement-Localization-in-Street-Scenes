# Object Placement Localization in Street Scenes
A candidate ranking framework with support-surface and hard constraints for text-guided object placement localization in street-scene images.

## Project Overview
This project predicts a plausible bounding box for placing an object in a street scene given a text prompt such as `place a person` or `place a car`.

## Main Idea
- Support-surface-guided candidate generation
- Semantic-aware candidate features
- Hard valid-mask constraints
- Candidate ranking instead of direct box regression

## Final Result
This framework achieved the best validation performance among our variants and produced more plausible person and car separation in qualitative visualization.

## Reference
This project is inspired by [BOOTPLACE (Hang Zhou)](https://github.com/RyanHangZhou/BOOTPLACE), but our setting is text-guided placement localization rather than full object composition. And we introduce support surface as the constraint to provide prior semantical information.