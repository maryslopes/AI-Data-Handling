# AI Data Handling Project M3 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.


#1. AI model selection
Refer to file "AI model selection.ipynb" for coding: https://github.com/maryslopes/AI-Data-Handling/blob/M3-project/AI%20model%20selection.ipynb

This project addresses a binary classification problem: predicting whether industrial equipment will experience machine failure based on operational sensor measurements. Two machine learning approaches were selected and compared: a Support Vector Machine (SVM) classifier and a feedforward Neural Network.

The SVM was chosen because it is a strong baseline model for structured tabular datasets and can capture non-linear relationships through the Radial Basis Function (RBF) kernel. Equipment failures are often influenced by complex interactions between temperature, rotational speed, torque, and tool wear, making a non-linear classifier appropriate.

The Neural Network was selected because it can automatically learn complex interactions between multiple sensor variables and has demonstrated strong performance in predictive maintenance applications. Although Neural Networks typically require longer training times than traditional machine learning algorithms, they can provide improved predictive performance when feature interactions are highly non-linear.

The objective of comparing both models was to identify the best trade-off between predictive accuracy and computational efficiency.


#2.Hyperparameter optimization
Refer to file "Hyperparameter.ipynb" for coding:  https://colab.research.google.com/github/maryslopes/AI-Data-Handling/blob/M3-project/Hyperparameter.ipynb#scrollTo=hn_SVk5JIeCV
Hyperparameter optimisation was performed to improve model performance while avoiding unnecessary model complexity.

For the SVM model, the following parameters were evaluated:

Kernel type (RBF)

Regularisation parameter C

Gamma parameter

Maximum number of iterations


The final configuration selected was:

kernel='rbf',
C=1.0,
gamma='scale',
max_iter=2000

This configuration achieved strong predictive performance while maintaining low training time.

For the Neural Network, the following parameters were investigated:

Hidden layer size
Learning rate
Batch size
Number of epochs
Dropout rate

The final Neural Network configuration was:
Hidden size = 128
Epochs = 50
Batch size = 32
Learning rate = 0.001
Dropout = 0.2

These values provided stable convergence and strong predictive performance while minimising overfitting.


#3.MLOps and experiment tracking

Experiments were tracked and compared systematically using version-controlled notebooks and cloud-hosted datasets stored in Google Cloud Storage. Each experiment recorded the preprocessing strategy, model configuration, training duration, and evaluation metrics.

The following experimental configurations were evaluated:

SVM with unscaled features
SVM with StandardScaler
SVM with MinMaxScaler
Neural Network with unscaled features
Neural Network with StandardScaler
Neural Network with MinMaxScaler
CPU versus GPU execution

This structured experimentation enabled reproducible comparison of model behaviour under different preprocessing and computational conditions.


#4.Analysis and conclusions from experiments

a. Preprocessing strategies
The experiments demonstrated that feature scaling had a substantial impact on model performance.

The unscaled dataset produced poor SVM performance (3.4% accuracy) and resulted in significantly higher Neural Network training loss. Both StandardScaler and MinMaxScaler improved optimisation and classification performance.

The highest Neural Network accuracy (97.5%) was achieved using StandardScaler, while the highest SVM accuracy (96.7%) was achieved using MinMaxScaler.

Although the Neural Network consistently achieved the highest predictive accuracy, the improvement over the SVM was relatively small. Consequently, the SVM remains an attractive option when computational efficiency is important.

Overall, the Neural Network was selected as the best-performing model due to its superior predictive accuracy. See results below:

Method	SVM Accuracy	NN Accuracy	SVM Training Time (s)	NN Training Time (s)
Unscaled	3.4%	96.6%	0.5906	33.4541
StandardScaler	91.8%	97.5%	0.8717	27.4825
MinMaxScaler	96.7%	97.3%	0.1730	27.8635
<img width="707" height="173" alt="image" src="https://github.com/user-attachments/assets/a6a16fb4-477b-4a02-b509-c6a85f3249d3" />

b. Learning Curve Plot
The learning curves demonstrate that feature scaling substantially improved neural network training. Models trained on standardised data showed a continuous reduction in training loss, whereas models trained on unscaled features exhibited persistently high loss values. This suggests that standardisation improved optimisation and convergence.

<img width="850" height="552" alt="image" src="https://github.com/user-attachments/assets/61ff1d91-4fd3-4bbb-96e2-cb9ed83b2d88" />

c. Misclassified Samples vs Feature Distributions
The largest spread of error points appears around Torque [Nm] and Tool wear [min]. hese variables may contribute more to prediction uncertainty.

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/a849c9b6-3b39-4788-89a0-35c91b144ac1" />


#5.Compute architecture selection

Neural Network experiments were executed using both CPU and GPU configurations.

Results showed:

Method	NN Accuracy	NN Training(Time (s)
StandardScaler CPU	97.5%	27.4825
StandardScaler GPU	97.8%	27.6330
<img width="515" height="88" alt="image" src="https://github.com/user-attachments/assets/60015f50-a6eb-4dcd-b55e-71f2d8d94272" />


Although GPU execution produced marginally higher accuracy, the training-time difference was negligible. Given the relatively small dataset size (10,000 records), GPU acceleration provided limited practical benefit. For larger predictive-maintenance datasets, GPU execution would likely offer greater advantages.

These findings demonstrate that compute architecture should be selected according to dataset size and training complexity rather than assumed benefits.


#6.Reproducibility 

The project environment was documented using a concise requirements.txt file containing the exact package versions required to reproduce the experiments.
Reproducibility was incorporated throughout the project.

Random seed fixed to 42 for consistent train/test splits.
Dataset stored in Google Cloud Storage.
Source code maintained in GitHub.
Dependencies documented using a concise requirements.txt file.
Hyperparameters explicitly recorded.
Training and evaluation performed using version-controlled notebooks.

These measures ensure that results can be reproduced and verified by other researchers.


#7.Model card / model documentation
Model Card
Intended Use

Predict machine failures from equipment sensor measurements to support predictive maintenance and improve manufacturing reliability.

Input Features
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Output

Binary classification:

0 = No machine failure
1 = Machine failure
Performance

Best Neural Network accuracy: 97.5%
Best SVM accuracy: 96.7%

Limitations: The model was trained on a historical dataset and may not fully generalise to different equipment types or operating conditions.


#8. Error analysis and model weaknesses
Misclassified samples were analysed to identify situations in which prediction errors occurred.

Error distributions were observed across all input variables but appeared more frequently in regions associated with higher torque and greater tool wear. This suggests that failure patterns within these operating conditions may be more difficult to distinguish.

Several limitations were identified:

The dataset contains a relatively small number of failure cases compared with non-failure cases.
Performance may decrease when deployed to equipment not represented in the training data.
Environmental and operational variables not captured in the dataset may influence machine failure.
The Neural Network requires substantially greater computational resources than the SVM.

Future work could focus on additional feature engineering, model calibration, ensemble methods, and collection of larger datasets containing more failure examples.


