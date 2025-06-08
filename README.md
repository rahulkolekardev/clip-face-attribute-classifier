
# Fine-tuned CLIP Model for Face Attribute Classification
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Page-yellow)](https://huggingface.co/syntheticbot/clip-face-attribute-classifier)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/syntheticbot/clip-face-attribute-classifier/colab)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://huggingface.co/syntheticbot/clip-face-attribute-classifier/kaggle)


This repository contains the model **`clip-face-attribute-classifier`**, a fine-tuned version of the **[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)** model. It has been adapted for multi-task classification of perceived age, gender, and race from facial images.

The model was trained on the **[FairFace dataset](https://github.com/joojs/fairface)**, which is designed to be balanced across these demographic categories. This model card provides a detailed look at its performance, limitations, and intended use to encourage responsible application.

## Model Description

The base model, CLIP (Contrastive Language-Image Pre-Training), learns rich visual representations by matching images to their corresponding text descriptions. This fine-tuned version repurposes the powerful vision encoder from CLIP for a specific classification task.

It takes an image as input and outputs three separate predictions for:
*   **Age:** 9 categories (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70)
*   **Gender:** 2 categories (Male, Female)
*   **Race:** 7 categories (White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino_Hispanic)

## Intended Uses & Limitations

This model is intended primarily for research and analysis purposes.

### Intended Uses
*   **Research on model fairness and bias:** Analyzing the model's performance differences across demographic groups.
*   **Providing a public baseline:** Serving as a starting point for researchers aiming to improve performance on these specific classification tasks.
*   **Educational purposes:** Demonstrating a multi-task fine-tuning approach on a vision model.

### Out-of-Scope and Prohibited Uses
This model makes predictions about sensitive demographic attributes and carries significant risks if misused. The following uses are explicitly out-of-scope and strongly discouraged:
*   **Surveillance, monitoring, or tracking of individuals.**
*   **Automated decision-making that impacts an individual's rights or opportunities** (e.g., loan applications, hiring decisions, insurance eligibility).
*   **Inferring or assigning an individual's self-identity.** The model's predictions are based on learned visual patterns and do not reflect how a person identifies.
*   **Creating or reinforcing harmful social stereotypes.**

## How to Get Started

To use this model, you need to import its custom `MultiTaskClipVisionModel` class, as it is not a standard `AutoModel`.

```python
import torch
from PIL import Image
from transformers import CLIPImageProcessor, AutoModel
import os
import torch.nn as nn

# --- 0. Define the Custom Model Class ---
# You must define the model architecture to load the weights into it.
class MultiTaskClipVisionModel(nn.Module):
    def __init__(self, num_labels):
        super(MultiTaskClipVisionModel, self).__init__()
        # Load the vision part of a CLIP model
        self.vision_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

        hidden_size = self.vision_model.config.hidden_size
        self.age_head = nn.Linear(hidden_size, num_labels['age'])
        self.gender_head = nn.Linear(hidden_size, num_labels['gender'])
        self.race_head = nn.Linear(hidden_size, num_labels['race'])

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return {
            'age': self.age_head(pooled_output),
            'gender': self.gender_head(pooled_output),
            'race': self.race_head(pooled_output),
        }

# --- 1. Configuration ---
MODEL_PATH = "syntheticbot/clip-face-attribute-classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Define Label Mappings (must match training) ---
age_labels = ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', 'more than 70']
gender_labels = ['Female', 'Male']
race_labels = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# Use sorted lists to create a consistent mapping
id_mappings = {
    'age': {i: label for i, label in enumerate(sorted(age_labels))},
    'gender': {i: label for i, label in enumerate(sorted(gender_labels))},
    'race': {i: label for i, label in enumerate(sorted(race_labels))},
}
NUM_LABELS = { 'age': len(age_labels), 'gender': len(gender_labels), 'race': len(race_labels) }

# --- 3. Load Model and Processor ---
processor = CLIPImageProcessor.from_pretrained(MODEL_PATH)
model = MultiTaskClipVisionModel(num_labels=NUM_LABELS)


model.to(DEVICE)
model.eval()

# --- 4. Prediction Function ---
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values=inputs['pixel_values'])

    predictions = {}
    for task in ['age', 'gender', 'race']:
        pred_id = torch.argmax(logits[task], dim=-1).item()
        pred_label = id_mappings[task][pred_id]
        predictions[task] = pred_label

    print(f"Predictions for {image_path}:")
    for task, label in predictions.items():
        print(f"  - {task.capitalize()}: {label}")
    return predictions

# --- 5. Run Prediction ---

predict('sample.jpg') # Replace with the path to your image
```

## Training Details

*   **Base Model:** [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
*   **Dataset:** [FairFace](https://github.com/joojs/fairface)
*   **Training Procedure:** The model was fine-tuned for 5 epochs. The vision encoder was mostly frozen, with only the final 3 transformer layers being unfrozen for training. A separate linear classification head was added for each task (age, gender, race). The total loss was the sum of the Cross-Entropy Loss from each of the three tasks.

## Evaluation

The model was evaluated on the FairFace validation split, which contains 10,954 images.

### Performance Metrics

The following reports detail the model's performance on each task.

#### **Gender Classification (Overall Accuracy: 96.38%)**
```
              precision    recall  f1-score   support

      Female       0.96      0.96      0.96      5162
        Male       0.96      0.97      0.97      5792

    accuracy                           0.96     10954
   macro avg       0.96      0.96      0.96     10954
weighted avg       0.96      0.96      0.96     10954
```

#### **Race Classification (Overall Accuracy: 73.22%)**
```
                 precision    recall  f1-score   support

          Black       0.90      0.89      0.89      1556
     East Asian       0.74      0.78      0.76      1550
         Indian       0.81      0.75      0.78      1516
Latino_Hispanic       0.58      0.62      0.60      1623
 Middle Eastern       0.69      0.57      0.62      1209
Southeast Asian       0.66      0.65      0.65      1415
          White       0.75      0.80      0.77      2085

       accuracy                           0.73     10954
      macro avg       0.73      0.72      0.73     10954
   weighted avg       0.73      0.73      0.73     10954
```

#### **Age Classification (Overall Accuracy: 59.17%)**
```
              precision    recall  f1-score   support

         0-2       0.93      0.45      0.60       199
       10-19       0.62      0.41      0.50      1181
       20-29       0.64      0.76      0.70      3300
         3-9       0.77      0.88      0.82      1356
       30-39       0.49      0.50      0.49      2330
       40-49       0.46      0.44      0.45      1353
       50-59       0.47      0.40      0.43       796
       60-69       0.45      0.32      0.38       321
more than 70       0.75      0.10      0.18       118

    accuracy                           0.59     10954
   macro avg       0.62      0.47      0.51     10954
weighted avg       0.59      0.59      0.58     10954
```

## Bias, Risks, and Limitations

*   **Perceptual vs. Identity:** The model predicts perceived attributes based on visual data. These predictions are not a determination of an individual's true self-identity.
*   **Performance Disparities:** The evaluation clearly shows that performance is not uniform across all categories. The model is significantly less accurate for certain racial groups (e.g., Latino_Hispanic, Middle Eastern) and older age groups. Using this model in any application will perpetuate these biases.
*   **Data Representation:** While trained on FairFace, a balanced dataset, the model may still reflect societal biases present in the original pre-training data of CLIP.
*   **Risk of Misclassification:** Any misclassification, particularly of sensitive attributes, can have negative social consequences. The model's moderate accuracy in age and race prediction makes this a significant risk.

### Citation

**Original CLIP Model:**
```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

**FairFace Dataset:**
```bibtex
@inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1548--1558},
  year={2021}
}
```
```
