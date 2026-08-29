# TSFR-Net

Official PyTorch implementation of **TSFR-Net: A Task-Specific Feature Routing Network for RGB-D Food Nutrition Estimation**.

TSFR-Net is an RGB-D food nutrition estimation framework designed to predict five meal-level targets: **calories, mass, fat, carbohydrate, and protein**. Instead of using the same fused representation for all regression targets, TSFR-Net focuses on **selective multi-scale feature organization and task-specific feature routing**.

The framework uses **DFormer-B** to extract interacted RGB-D multi-scale features, followed by:

* **Dual-path multi-scale feature refinement** for complementary hierarchical feature learning.
* **Selected shallow–deep representations** to preserve local details and high-level semantic information while reducing redundant all-scale aggregation.
* **Task-specific feature routing**, which assigns different feature branches to different nutrition targets.
* **Lightweight residual correction** for further refinement of mass and carbohydrate predictions.

## Dataset

Experiments are conducted on the **Nutrition5K** RGB-D dataset. The model takes paired RGB and depth images as input and directly predicts five nutritional quantities.

## Results

Under the single-model setting **without data augmentation or external semantic information**, TSFR-Net achieves the following PMAE results:

| Target       | PMAE (%) ↓ |
| ------------ | ---------: |
| Calories     |      13.97 |
| Mass         |      11.02 |
| Fat          |      19.52 |
| Carbohydrate |      20.58 |
| Protein      |      19.27 |
| **Mean**     |  **16.87** |

## Citation

If you find this work useful, please consider citing our paper. Citation information will be updated after publication.

## Acknowledgement

This implementation uses **DFormer-B** as the RGB-D feature extraction backbone. We thank the authors of DFormer and Nutrition5K for their contributions to the research community.
