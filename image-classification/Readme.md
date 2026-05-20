# Stanford Cars Image Classification

## 1. Project Overview

This project focuses on image classification using the Stanford Cars dataset. The main goal was not only to train a model, but also to modify CNN architectures and compare the effect of these changes.


The project includes custom CNN models and transfer learning experiments. The main experiments are:

- Baseline AlexNet-like CNN
- CarNet v1
- CarNet v2
- CarNet v3
- ResNet18 transfer learning on 30 classes
- ResNet18 Full196 transfer learning on all 196 classes

The main objective was to observe how architectural changes, training strategies, and pretrained models affect image classification performance.

---

## 2. Dataset

The dataset used in this project is the Stanford Cars dataset.

The dataset contains 196 car classes. Each class represents a specific car model and year. This makes the task a fine-grained image classification problem because many classes are visually similar.

Example class names include:

Acura Integra Type R 2001
Aston Martin V8 Vantage Coupe 2012
Audi TT RS Coupe 2012
BMW X5 SUV 2007
Chevrolet Corvette Convertible 2012

Dataset structure expected by the scripts:

data/
└── stanford_cars/
    ├── train/
    └── test/

The dataset is not included in this repository because of file size. It should be downloaded separately and placed in the expected folder structure.

3. Problem Definition

The task is multi-class vehicle image classification.

For the first experiments, a 30-class subset was used to allow faster experimentation. Later, the full 196-class Stanford Cars problem was used with ResNet18 transfer learning.

The main evaluation metrics are:

Accuracy
Macro-F1
Weighted-F1
Classification Report
Confusion Matrix
Top-5 Accuracy for the 196-class experiment
4. Baseline CNN / AlexNet-like Model

The first model was a simple CNN / AlexNet-like baseline. This model was used as a starting point.

A simple CNN generally consists of:

Convolution
ReLU
Max Pooling
Flatten
Fully Connected Classifier

This baseline model produced low performance on the Stanford Cars dataset.

Model	Test Accuracy	Macro-F1
Baseline AlexNet-like CNN	0.0729	0.0204

This result showed that a simple CNN was not sufficient for fine-grained car classification. Stanford Cars contains visually similar vehicle classes, so the model needs stronger feature extraction and better generalization.

5. CarNet v1

CarNet v1 was designed as an improved CNN architecture compared with the baseline.

The main additions were:

Batch Normalization
Residual connections
SE channel attention mechanism
Global Average Pooling
Dropout
5.1. Batch Normalization

Batch Normalization normalizes the output of convolution layers. It helps stabilize training by keeping activation distributions more consistent.

In the model, the structure is:

Conv2D → BatchNorm → ReLU

This is different from a simple CNN block, which usually uses:

Conv2D → ReLU
5.2. Residual Connections

Residual connections add the input of a block directly to its output.

Instead of learning only:

x → Conv → Conv → output

the model learns:

x → Conv → Conv → output + x

This helps preserve information and improves gradient flow.

5.3. SE Channel Attention

SE means Squeeze-and-Excitation. It is a channel attention mechanism.

CNN feature maps contain many channels. Some channels may be more important for classification than others. The SE block learns channel weights and emphasizes more useful feature channels.

The basic idea is:

feature map
→ global channel summary
→ channel importance weights
→ weighted feature map
5.4. Global Average Pooling

Instead of using a very large fully connected structure, Global Average Pooling summarizes each feature channel into one value.

This reduces the number of parameters and helps reduce overfitting.

5.5. Dropout

Dropout randomly disables some neurons during training. This prevents the model from depending too much on a small set of neurons and helps reduce overfitting.

CarNet v1 improved over the baseline.

Model	Test Accuracy	Macro-F1
Baseline AlexNet-like CNN	0.0729	0.0204
CarNet v1	0.1360	0.0905
6. CarNet v2

CarNet v2 improved the architecture and training strategy.

The main changes were:

Stronger data augmentation
AdamW optimizer
Weight decay
Cosine learning rate scheduler
Label smoothing
RandomErasing
Longer training
6.1. Data Augmentation

Data augmentation was used to make the model more robust.

The main augmentation techniques were:

Technique	Purpose
Resize	Standardizes image size
RandomCrop	Reduces location dependency
RandomHorizontalFlip	Improves robustness to vehicle direction
RandomAffine	Adds small rotation, translation, scale, and shear
ColorJitter	Adds brightness, contrast, saturation, and hue variation
RandomErasing	Prevents dependency on a single visual region
Normalize	Normalizes pixel values

The augmentation values were kept controlled. This is important because car classification depends on small visual details such as headlights, wheels, body shape, and front grille. Very aggressive augmentation could damage these details.

6.2. AdamW Optimizer

AdamW was used as the optimizer. AdamW is a version of Adam that applies weight decay in a more correct way.

AdamW helps with stable optimization and is commonly used in modern deep learning training.

6.3. Weight Decay

Weight decay is a regularization technique. It prevents model weights from becoming too large.

This helps reduce overfitting by encouraging the model to learn simpler and more generalizable patterns.

6.4. Cosine Scheduler

CosineAnnealingLR was used to gradually reduce the learning rate during training.

The idea is:

higher learning rate at the beginning
lower learning rate near the end

This allows the model to learn faster at first and then make smaller updates later.

6.5. Label Smoothing

Label smoothing reduces overconfidence by preventing the model from assigning all probability to a single class during training.

This can improve generalization and make the model less overconfident.

6.6. CarNet v2 Results

CarNet v2 produced better results than previous custom CNN models.

Model	Test Accuracy	Macro-F1
CarNet v2 short training	0.1982	0.1643
CarNet v2 with 60 epochs	~0.45	~0.45

The 60-epoch result showed that the model was still learning and needed longer training to reach better performance.

7. CarNet v3

CarNet v3 tested a more complex architecture.

The main idea was to use a richer representation by combining different pooling outputs, such as:

Global Average Pooling
Global Max Pooling
Feature fusion

However, the result was not better than CarNet v2.

Model	Test Accuracy	Macro-F1
CarNet v3	0.1704	0.1328

This showed that increasing architectural complexity does not always improve performance. A more complex model can be harder to optimize or may overfit when the dataset size is limited.

8. ResNet18 Transfer Learning

After testing custom CNN architectures, ResNet18 transfer learning was used.

ResNet18 is a pretrained convolutional neural network. It was originally trained on ImageNet. Instead of starting from random weights, the model starts with already learned visual features.

This is useful because early layers of CNNs learn general image features such as:

edges
colors
textures
basic shapes

These features can be useful for car classification as well.

9. ResNet18 30-Class Experiment

For the 30-class experiment, ResNet18 was adapted to the selected Stanford Cars subset.

The original ImageNet classifier was replaced with a classifier suitable for the number of car classes.

The training followed transfer learning logic:

Pretrained ResNet18 backbone
→ Replace final classifier
→ Train classifier and selected layers
→ Evaluate on car classes

Results:

Model	Test Accuracy	Macro-F1
ResNet18 Transfer Learning	0.7330	0.7315
ResNet18 Fusion Head	0.6683	0.6438

The standard ResNet18 transfer learning approach performed better than the fusion head experiment.

10. ResNet18 Full196 v2

The final and strongest image classification experiment used all 196 Stanford Cars classes.

This version used a staged fine-tuning strategy.

10.1. Classifier

The classifier is the final part of the model that converts extracted image features into class predictions.

ResNet18 normally has a final layer for 1000 ImageNet classes:

512 → 1000

For Stanford Cars, this was replaced with a new classifier for 196 classes:

512 → hidden layer → 196
10.2. Stage 1

In Stage 1, only the classifier was trainable.

Backbone frozen
Classifier trainable

This allows the new classifier to learn the Stanford Cars classes without changing the pretrained feature extractor.

10.3. Stage 2

In Stage 2, layer4 and the classifier were trainable.

layer4 + classifier trainable

layer4 is the final residual block group of ResNet18. It learns higher-level and more class-specific features.

10.4. Stage 3

In Stage 3, layer3, layer4, and the classifier were trainable.

layer3 + layer4 + classifier trainable

This allows deeper adaptation to the car classification task.

The learning rate was reduced in later stages to avoid damaging the pretrained weights.

11. ResNet18 Layers

ResNet18 consists of:

conv1
bn1
relu
maxpool
layer1
layer2
layer3
layer4
avgpool
fc / classifier

The earlier layers detect general low-level features, while later layers detect more abstract and task-specific features.

Part	General Meaning
conv1 / layer1	edges, colors, basic lines
layer2	textures and shape parts
layer3	object parts such as lights, wheels, body shape
layer4	more abstract class-related features
classifier	final class decision

As the model goes deeper, the feature map becomes smaller, but each activation represents a larger area of the original image.

12. Top-5 Accuracy

Top-5 Accuracy measures whether the correct class is among the model’s top 5 predictions.

This is useful for a 196-class problem because many car classes are visually similar.

For example:

Top-1 prediction: wrong
Correct class appears in top 5 predictions: counted as correct for Top-5

The final ResNet18 Full196 v2 result:

Model	Test Accuracy	Macro-F1	Top-5 Accuracy
ResNet18 Full196 v2	0.5811	0.5666	0.8664

The Top-5 value shows that the model often includes the correct car class among its most likely predictions.

13. Final Image Classification Results
Model	Dataset	Accuracy	Macro-F1	Top-5
Baseline AlexNet-like	30 classes	0.0729	0.0204	-
CarNet v1	30 classes	0.1360	0.0905	-
CarNet v2 short training	30 classes	0.1982	0.1643	-
CarNet v2 60 epochs	30 classes	~0.45	~0.45	-
CarNet v3	30 classes	0.1704	0.1328	-
ResNet18 Transfer	30 classes	0.7330	0.7315	-
ResNet18 Fusion Head	30 classes	0.6683	0.6438	-
ResNet18 Full196 v1	196 classes	0.2890	0.2488	0.6140
ResNet18 Full196 v2	196 classes	0.5811	0.5666	0.8664
14. Output Files

The outputs/ folder contains training and evaluation outputs such as:

classification reports
confusion matrices
training curves
augmentation examples
feature map visualizations
model summaries
metrics files
experiment results

The trained model weight files are not included in the repository.

15. How to Run

Install dependencies:

pip install -r requirements.txt

Prepare the dataset in the expected folder structure:

data/
└── stanford_cars/
    ├── train/
    └── test/

Run selected experiments:

python src/train_carnet_v2.py
python src/train_resnet18_full196_v2.py
16. Conclusion

This project showed the difference between custom CNN design and transfer learning.

The custom CarNet models improved over the baseline CNN, especially CarNet v2 with longer training. However, the best performance was achieved with ResNet18 transfer learning.

The results show that pretrained models provide a strong advantage for fine-grained image classification tasks such as Stanford Cars.
