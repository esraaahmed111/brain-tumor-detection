# Brain Tumor Detection and Segmentation

Machine learning pipeline for brain tumor classification and pixel-level
segmentation from MRI scans using PyTorch.

## Models

| Task           | Architecture | Dataset                                  |
|----------------|--------------|------------------------------------------|
| Classification | ResNet50     | 506 labeled MRI images (yes/no tumor)    |
| Segmentation   | U-Net        | LGG MRI, 110 patients, 3929 paired masks |

## Project Structure

```
brain-tumor-detection/
├── main.py
├── classifier.py
├── segmentation.py
├── app.py
├── models/
│   └── unet.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Datasets

- [Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
