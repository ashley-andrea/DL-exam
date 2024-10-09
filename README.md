# Image Segmentation Using U-Net Inspired CNN
-----
### Project Overview
This project focuses on performing image segmentation using deep learning techniques. The dataset used consists of aerial images, where the task is to classify each pixel into one of six possible categories. The primary aim was to explore various deep learning tools, rather than optimize for the best possible score, and to implement an encoder-decoder convolutional neural network (CNN) architecture inspired by the U-Net structure.

### Dataset
The dataset contains aerial images and their corresponding masks. Each image is segmented into six classes, where each pixel is assigned a class label. The dataset is large, so **Git Large File Storage (Git LFS)** was used to manage the size of the files, and you may need to configure Git LFS to access the full dataset.

### Architecture: U-Net Inspired CNN
The model architecture was based on the U-Net structure, which is highly effective for image segmentation tasks. Key aspects of the architecture include:

- **Encoder-Decoder Structure:** The model uses a contracting path to capture context (encoder) and an expansive path (decoder) to enable precise localization.
- **Skip Connections:** These connections help forward and backward passes by mitigating potential gradient issues such as exploding or vanishing gradients.
- **Distributed 3x3 Convolutions:** Used to increase the receptive field and allow the network to capture more context from the image.
- **Max Pooling:** Applied in the contracting path for downsampling the feature maps.
- **Transpose Convolutions:** Used in the expansive path for upsampling the feature maps to match the input size.
- **'Same' Padding:** Ensures that the input and output sizes remain coherent across layers.

### Model Details
- **Input Size:** 256x256 pixels per image, with corresponding segmentation masks of the same size.
- **Preprocessing:** Pixel values were rescaled from the original [0, 255] range to [0, 1].
- **Activation Functions:**
  - **ReLU:** Used in almost all layers for non-linearity.
  - **Softmax:** Applied in the output layer to achieve pixel-wise multi-class classification.
- **Loss Function:** A sparse categorical cross-entropy loss was used, suitable for multi-class classification with integer class labels.
- **Custom Function for Mask Processing:** Since the masks were of shape `256x256x1`, a function was created to separate and handle the classes for comparison between predicted outputs and desired outputs.

### Model Training
- **Evaluation Metrics:** Given the presence of imbalanced classes, evaluation extended beyond simple accuracy to consider additional metrics such as precision, recall, and F1-score, providing a more comprehensive assessment.
- **Cross-Validation:** Cross-validation was performed to tune hyperparameters and identify the optimal configuration for the network.
- **Hyperparameter Tuning:** This involved adjusting the learning rate, the number of filters, and network depth.
- **Resource Limitations:** Due to computational limitations in Google Colab, several adjustments were made to make the model runnable:
  - The dataset was halved.
  - The number of filters was reduced.
  - The depth of the network was shortened.
  However, these trade-offs can be reversed if more resources are available to achieve better performance.

### Results
The primary focus of this project was not on maximizing accuracy but rather on experimenting with and implementing various deep learning techniques. Despite the reduced model complexity due to resource constraints, the model demonstrated satisfactory performance for the task. The final model was retrained on the entire dataset after hyperparameter tuning.

### Important!!
**Git LFS** was used to manage the large dataset, which **may affect the visibility** of files in GitHub. If you're cloning or working with this project, make sure to install and configure Git LFS properly to access the dataset.
