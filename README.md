Info-6147: FER2013 Facial Expression Classification

Final PyTorch project for classifying facial expressions using the FER2013 dataset. This project trains a convolutional neural network (CNN) to recognize 7 facial emotions.


Dataset: FER2013

Training Size: ~30,000, Test: ~3,500

Grayscale facial images of size 48x48

7 emotion classes:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral


Model Architecture

class EmotionCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


Training

Optimizer: Adam or SGD

Loss Function: CrossEntropyLoss

Input: normalized tensors from 48x48 grayscale images

Data Augmentation: optional with transforms

Regularization: dropout


Hyperparameter Tuning

Learning Rate: [0.001, 0.0005]

Dropout Rate: [0.3, 0.5]

Optimizers: adam, sgd

Batch Size: 64

Each config trained for 5 epochs. Best model parameters are saved.


Evaluation

Accuracies

Class-wise performance using confusion matrix

Visualize model predictions on example images


Visualizations

Loss and Accuracy Plot per epoch

Confusion Matrix using sklearn