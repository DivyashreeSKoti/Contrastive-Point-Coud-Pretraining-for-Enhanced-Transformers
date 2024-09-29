# Contrastive Pretrained Set Transformer (CPST) and Contrastive Point Cloud Transformer (CPCT)
## Pointcloud processing for 3D Body Scan, ShapeNet, and ModelNet dataset
This repository contains code and models for performing classification and Regression tasks on 3D datasets, including 3D Body Scan, ModelNet, and ShapeNet. This project explores the Simple Framework for Contrastive Learning (SimCLR) technique integrated with Set Transformer (ST) and Point Cloud Transformer (PCT) to create the pretrained models: Contrastive Pretrained Set Transformer (CPST) and Contrastive Point Cloud Transformer (CPCT) for various tasks of the datasets.

Objective:
1. Analyzing the potential advantages and challenges of Contrastive Pretrained Set Transformer (CPST) and Contrastive Point Cloud Transformer (CPCT).
2. A comparative study between the Naive and the corresponding Contrastive pretrained models.
3. Assess the generalization capabilities of both models in their Naive and Contrastive.

Dataset:
3D body scan dataset is available at https://github.com/DivyashreeSKoti/BodyScan_Data.git
(Note: For ShapeNet and ModelNet please refer to paper)

3D body scan example:
![image](https://github.com/user-attachments/assets/2add6c68-50e5-4462-a3e3-92b74de4fe70)

Weak generalization representation:
![image](https://github.com/user-attachments/assets/bc1d8b09-12a1-4dc2-83eb-830d1420b742)

Tasks include both Weak and Strong generalization:
1. Self Identification of 3D body scans
2. Gender classification
3. Binned age classification
4. Regressions tasks (predicting height, weight and age)
5. Shapenet object classification
6. Modelnet object classification.







