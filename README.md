# Object Placement Localization in Street Scenes  

***Object Placement Localization in Street Scenes*** is a project that predicts where a new object, such as a person or a car, can be plausibly placed in a street-scene image given a text prompt. Instead of directly regressing one bounding box, our method first prepares a set of semantically reasonable candidate boxes, then uses a ranking neural network to score them, and finally selects the best placement. The overall pipeline includes support-surface-guided candidate preparation, candidate ranking, and top-1 box selection.  
![workflow](./assets/workflow.png)

## Installation  
- **Environment:**  
    - Python version 3.10 or higher is required.
    - Install the dependencies:  
    `pip install -r requirements.txt`
- **Dataset:**  
    - we preprocess the [Cityscapes dataset](https://www.cityscapes-dataset.com/) to construct a **BOOTPLACE-like dataset ([Download](https://ualberta-my.sharepoint.com/personal/zchai3_ualberta_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzchai3%5Fualberta%5Fca%2FDocuments%2FECE740%2Dgroup3%2Ddataset&ga=1))** for training and evaluating this framework.
    - Please place the downloaded dataset folder under the project root directory as `./bootplace_like_data/`

## How to use  
```
project_root/
├── placement_localization_pipeline.ipynb
├── requirements.txt
└── bootplace_like_data/  
```
After completing the installation, place the dataset under the project root directory. The main code is provided in `placement_localization_pipeline.ipynb`, which contains clear outlines and notes for each part of the pipeline. Open this notebook and run the cells sequentially.

## Reference

This project is inspired by [BOOTPLACE (Hang Zhou)](https://github.com/RyanHangZhou/BOOTPLACE), but our setting is text-guided placement localization rather than full object composition. And we introduce support surface as the constraint to provide prior semantical information.
