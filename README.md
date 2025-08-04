# RescueVision

Developed an automated human detection system for search and rescue drone imagery using YOLOv11 computer vision technology, achieving 85% mean Average Precision (mAP) through advanced machine learning techniques and comprehensive bias mitigation strategies, all within AI4ALL's innovative AI4ALL Ignite accelerator program.

---

## Problem Statement 

Rapid and accurate human detection in aerial images is crucial for search and rescue (SAR) operations, where every minute can be the difference between life and death. Traditional manual analysis of drone footage is time-consuming, error-prone, and creates dangerous delays during emergency situations. The lack of automated detection systems forces SAR teams to rely on labor-intensive manual processes that can delay life-saving interventions and waste critical resources during search operations.

---

## Key Results 

1. Achieved 85% mean Average Precision (mAP) on wilderness drone imagery human detection
2. Processed 4,000+ training images from the SARD dataset with comprehensive data augmentation
3. Identified and addressed three critical bias sources in search and rescue AI systems:
   
   - Actor demographic representation bias affecting detection accuracy across skin tones
   - Environmental context bias limits performance to wilderness settings
   - Weather and lighting condition bias reducing robustness in adverse scenarios
     
5. Implemented real-time detection capability suitable for emergency response time constraints
6. Developed a comprehensive evaluation framework measuring precision, recall, F1-score, and false-positive rates
7. Created bias mitigation strategies through data augmentation and demographic analysis

---

## Methodologies 

To accomplish this, we implemented YOLOv11 object detection architecture with transfer learning from pre-trained weights, utilizing Google Colab for computational resources and systematic hyperparameter optimization. The model was trained on the SARD dataset with comprehensive data preprocessing, augmentation techniques, and rigorous evaluation protocols. We engineered a complete machine learning pipeline incorporating bias analysis, performance visualization, and robust testing across diverse environmental conditions. Through iterative training cycles and validation testing, we achieved our target 85% mAP while implementing fairness measures to ensure equitable performance across demographic groups and environmental scenarios.

---

## Data Sources 

Kaggle Dataset: [SARD - Search And Rescue Dataset](https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue)

---

## Technologies Used 

- **Python - Primary programming language for model development**
- **YOLOv11 - State-of-the-art object detection architecture**
- **Google Colab - Cloud-based training environment**
- **Roboflow - Dataset management and preprocessing**
- **GitHub - Version control and project collaboration**
- **Ultralytics - YOLO model implementation library**

---

## Authors 

This project was completed in collaboration with:
- [@DLiu7](https://github.com/DLiu7)
- [@ankushachwani](https://github.com/ankushachwani)
- [@NER2160349](https://github.com/NER2160349)
- [@Shay-7278](https://github.com/Shay-7278)
- [@Mahir-MShahriar](https://github.com/Mahir-MShahriar)
