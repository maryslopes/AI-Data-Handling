# AI Data Handling Project M3 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.


##1. AI model selection


Refer to file "AI model selection.ipynb" for coding: https://github.com/maryslopes/AI-Data-Handling/blob/M3-project/AI%20model%20selection.ipynb

This project addresses a binary classification problem: predicting whether industrial equipment will experience machine failure based on operational sensor measurements. Two machine learning approaches were selected and compared: a Support Vector Machine (SVM) classifier and a feedforward Neural Network.

The SVM was chosen because it is a strong baseline model for structured tabular datasets and can capture non-linear relationships through the Radial Basis Function (RBF) kernel. Equipment failures are often influenced by complex interactions between temperature, rotational speed, torque, and tool wear, making a non-linear classifier appropriate.

The Neural Network was selected because it can automatically learn complex interactions between multiple sensor variables and has demonstrated strong performance in predictive maintenance applications. Although Neural Networks typically require longer training times than traditional machine learning algorithms, they can provide improved predictive performance when feature interactions are highly non-linear.

The objective of comparing both models was to identify the best trade-off between predictive accuracy and computational efficiency.

The original dataset contained 15 variables. Feature selection was performed to retain only variables expected to influence machine degradation and failure. The selected features were machine type, air temperature, process temperature, rotational speed, torque and tool wear. Variables representing specific failure categories and identifiers were excluded because they either contained target-related information or provided little predictive value. Machine type was one-hot encoded, producing a final set of seven preprocessed features used for model training.

##2.Hyperparameter Optimization


Refer to file "Hyperparameter.ipynb" for coding:  https://colab.research.google.com/github/maryslopes/AI-Data-Handling/blob/M3-project/Hyperparameter.ipynb#scrollTo=hn_SVk5JIeCV


Hyperparameter optimisation was performed to improve model performance while avoiding unnecessary model complexity.

For the SVM model, different values of the regularisation parameter C and gamma were evaluated. Performance was assessed using validation accuracy and training time. The final configuration (kernel='rbf', C=1.0, gamma='scale', max_iter=2000) provided the best balance between predictive performance and computational efficiency.


This configuration achieved strong predictive performance while maintaining low training time.

For the Neural Network, the following parameters were investigated:

Hidden layer size
Learning rate
Batch size
Number of epochs
Dropout rate

The final Neural Network configuration was:

Hidden size = 100

Epochs = 50

Batch size = 32

Learning rate = 0.01

Dropout = 0.2

These values provided stable convergence and strong predictive performance while minimising overfitting.


##3.MLOps and experiment tracking


Experiments were tracked and compared systematically using version-controlled notebooks and cloud-hosted datasets stored in Google Cloud Storage. Each experiment recorded the preprocessing strategy, model configuration, training duration, and evaluation metrics.
Each experiment was uniquely identified by model type, preprocessing strategy, hidden-layer size, learning rate and execution environment. TensorBoard logs and notebook outputs were used to compare experimental runs and support model-selection decisions.

The following experimental configurations were evaluated:

SVM with unscaled features

SVM with StandardScaler

SVM with MinMaxScaler

Neural Network with unscaled features

Neural Network with StandardScaler

Neural Network with MinMaxScaler

CPU versus GPU execution

This structured experimentation enabled reproducible comparison of model behaviour under different preprocessing and computational conditions.

TensorBoard was used to track Neural Network experiments and compare different hyperparameter configurations. Validation accuracy, training duration and convergence behaviour were monitored for each run, enabling systematic comparison of candidate models.


##4.Analysis and conclusions from experiments


###a. Preprocessing strategies


The experiments demonstrated that feature scaling had a substantial impact on model performance.

The unscaled dataset produced poor SVM performance (3.4% accuracy) and resulted in significantly higher Neural Network training loss. 
The poor performance of the unscaled SVM (3.4% accuracy) highlights the sensitivity of distance-based classifiers to feature scale. This result further demonstrates the importance of feature normalisation when applying SVM models to industrial sensor data.
Both StandardScaler and MinMaxScaler improved optimisation and classification performance.

The highest Neural Network accuracy (97.5%) was achieved using StandardScaler, while the highest SVM accuracy (96.7%) was achieved using MinMaxScaler.

Although the Neural Network consistently achieved the highest predictive accuracy, the improvement over the SVM was relatively small. Consequently, the SVM remains an attractive option when computational efficiency is important.

Overall, the Neural Network was selected as the best-performing model due to its superior predictive accuracy. See results below:


| Method | SVM Accuracy | NN Accuracy | SVM Training Time (s) | NN Training Time (s) |
|----------|----------|----------|----------|----------|
| Unscaled | 3.4% | 96.6% | 0.5906 | 33.4541 |
| StandardScaler | 91.8% | 97.5% | 0.8717 | 27.4825 |
| MinMaxScaler | 96.7% | 97.3% | 0.1730 | 27.8635 |

###b. Learning Curve Plot


The learning curves demonstrate that feature scaling substantially improved neural network training. Models trained on standardised data showed a continuous reduction in training loss, whereas models trained on unscaled features exhibited persistently high loss values. This suggests that standardisation improved optimisation and convergence.

<img width="850" height="552" alt="image" src="https://github.com/user-attachments/assets/61ff1d91-4fd3-4bbb-96e2-cb9ed83b2d88" />

###c. Misclassified Samples vs Feature Distributions
The largest spread of error points appears around Torque [Nm] and Tool wear [min]. These variables may contribute more to prediction uncertainty.

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/a849c9b6-3b39-4788-89a0-35c91b144ac1" />

Based on the experimental results, the Neural Network trained using StandardScaler preprocessing was selected as the preferred model for deployment. This model achieved the highest predictive accuracy (97.5%) while maintaining stable convergence throughout training. Although the SVM offered superior computational efficiency, the additional predictive performance provided by the Neural Network was considered more valuable for early detection of equipment failures within pharmaceutical manufacturing environments.


###d. Neural Network Hyperparameter Search

TensorBoard was used to track Neural Network experiments and visualise both training loss and validation accuracy. This enabled systematic comparison of hyperparameter combinations and provided evidence for selecting the final model configuration.
TensorBoard was used to monitor training loss across multiple Neural Network configurations. Hidden-layer sizes of 100, 500 and 1000 neurons were evaluated together with learning rates of 0.001, 0.01 and 0.1.


The highest validation accuracy (98.0%) was achieved using a hidden-layer size of 100 neurons and a learning rate of 0.01.

Larger neural network architectures did not significantly improve performance and resulted in substantially longer training times. For example, networks using 1000 hidden neurons required several minutes of training while providing only marginal accuracy improvements compared with smaller architectures.

These results suggest that a relatively compact neural network was sufficient for the equipment-failure prediction task and provided the best trade-off between predictive performance and computational efficiency.

<img width="1055" height="496" alt="image" src="https://github.com/user-attachments/assets/431ca629-92d2-4073-9260-221cb610a6c5" />

In terms of Training Loss analysis, the results demonstrated that learning rate had a substantial impact on model convergence. Models trained with a learning rate of 0.01 achieved the lowest training losses and the highest validation accuracies. The best-performing configuration (hidden size = 100, learning rate = 0.01) achieved a final training loss of approximately 0.049 and a validation accuracy of 98%.

By contrast, configurations using a learning rate of 0.1 failed to converge and maintained a loss of approximately 3.39 throughout training. This indicates that the learning rate was too large and prevented effective optimisation.

Increasing the hidden-layer size from 100 to 500 or 1000 neurons did not produce meaningful improvements in predictive performance while substantially increasing training time. Consequently, a smaller architecture was preferred because it achieved comparable or better accuracy with significantly lower computational cost.

These findings demonstrate the importance of hyperparameter optimisation and support the final Neural Network configuration selected for this project.
<img width="1026" height="495" alt="image" src="https://github.com/user-attachments/assets/4b04bb1b-5473-4862-925a-8eb14d361367" />


##5.Compute architecture selection

Neural Network experiments were executed using both CPU and GPU configurations.

Results showed:

| Method | NN Accuracy | NN Training Time (s) |
|----------|----------|----------|
| StandardScaler CPU | 97.5% | 27.4825 |
| StandardScaler GPU | 97.8% | 27.6330 |


Although GPU execution produced marginally higher accuracy, the training-time difference was negligible. Given the relatively small dataset size (10,000 records), GPU acceleration provided limited practical benefit. For larger predictive-maintenance datasets, GPU execution would likely offer greater advantages.

These findings demonstrate that compute architecture should be selected according to dataset size and training complexity rather than assumed benefits.


##6.Reproducibility 


The project environment was documented using a concise requirements.txt file containing the exact package versions required to reproduce the experiments.
Reproducibility was incorporated throughout the project.

Random seed fixed to 42 for consistent train/test splits.
Dataset stored in Google Cloud Storage.
Source code maintained in GitHub.
Dependencies documented using a concise requirements.txt file.
Hyperparameters explicitly recorded.
Training and evaluation performed using version-controlled notebooks.

These measures ensure that results can be reproduced and verified by other researchers.


##7.Model card / model documentation


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


##8. Error analysis and model weaknesses


Misclassified samples were analysed to identify situations in which prediction errors occurred.

Error distributions were observed across all input variables but appeared more frequently in regions associated with higher torque and greater tool wear. This suggests that failure patterns within these operating conditions may be more difficult to distinguish.

Several limitations were identified:

The dataset contains a relatively small number of failure cases compared with non-failure cases.
Performance may decrease when deployed to equipment not represented in the training data.
Environmental and operational variables not captured in the dataset may influence machine failure.
The Neural Network requires substantially greater computational resources than the SVM.


## 9. Final Recommendation

Based on all experiments conducted, the Neural Network trained on StandardScaler-preprocessed features was selected as the preferred model for deployment. This configuration achieved the highest predictive accuracy (97.5%), stable convergence behaviour and strong validation performance. Although the SVM offered substantially faster training times, the Neural Network's superior predictive capability was considered more valuable for early identification of equipment failures in pharmaceutical manufacturing environments.

The project demonstrates that careful feature engineering, preprocessing, hyperparameter optimisation and experiment tracking can significantly improve predictive-maintenance performance while maintaining reproducibility.




