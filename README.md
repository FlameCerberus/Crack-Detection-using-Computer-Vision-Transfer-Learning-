# Crack Detection using MobileNetV2 and Transfer Learning

## Description
This project aims to develop a crack detection system using deep learning techniques, specifically leveraging transfer learning with the MobileNetV2 architecture. The detection of cracks in infrastructure such as roads, buildings, and bridges is crucial for maintenance and safety purposes. By automating this process through machine learning, we can efficiently identify areas that require attention, potentially preventing accidents and saving resources.

## Data Cleaning Process
The dataset used for this project consists of images containing both cracked and intact surfaces. Prior to training the model, it's essential to preprocess the data to ensure consistency and quality. The data cleaning process involves:
- Resizing images to a standard size compatible with the MobileNetV2 input dimensions.
- Normalizing pixel values to improve convergence during training by using preprocessing in mobilevnet2.
- Augmentation was not required as the dataset was large enough and the model was fed with enough data for it to train properly
![PICTURESEXMP](https://github.com/FlameCerberus/Crack-Detection-using-Computer-Vision-Transfer-Learning-/assets/96816249/dbc612f5-633b-4a6f-bd4d-d28b7bed377b)

## Training The Model by Transfer Learning
Transfer learning involves leveraging pre-trained models trained on large datasets and fine-tuning them for a specific task. In this project, MobileNetV2 architecture was utilized and the architecture pre-trained. By freezing the initial layers and replacing the final classification layers, we adapt the network to the crack detection task. During training, we optimize the model parameters using a suitable loss using binary crossentropy as we are doing 1 or 0 clasification, function and optimizer adam.

## Fine-tuning the Model Further
After the initial training, we can fine-tune the model by unfreezing some of the earlier layers and retraining them with a lower learning rate. Fine-tuning allows the model to learn task-specific features better and improve its performance on the crack detection task. THe finetuning was done by unfreezing the top 10 layers of the transfered model
![FineTuning](https://github.com/FlameCerberus/Crack-Detection-using-Computer-Vision-Transfer-Learning-/assets/96816249/29196667-c136-46a6-a77b-7e1e1f22b7cc)

## Model Evaluation Part
To assess the performance of the crack detection model, a model test evaluation was ran and gave a very promising result in detecting whether the image shows cracked or not cracked situation
![Model_Evaluation_test](https://github.com/FlameCerberus/Crack-Detection-using-Computer-Vision-Transfer-Learning-/assets/96816249/b89ae82b-c277-4e14-a734-e70fba86dda2)

## Conclusion
In conclusion, this project demonstrates the effectiveness of transfer learning using the MobileNetV2 architecture for crack detection. By leveraging pre-trained models and fine-tuning them on a specific task, we can develop accurate and efficient detection systems. The ability to automate crack detection has significant implications for infrastructure maintenance and safety. MobileNetV2 is a very robust model as it was able to detect the test data at full accuracy.

## Reference
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Cracked or not-Cracked Mendeley Dataset](https://data.mendeley.com/public-files/datasets/5y9wdsg2zt/files/8a70d8a5-bce9-4291-bab9-b48cfb3e87c3/file_downloaded)
