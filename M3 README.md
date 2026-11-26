# AI Data Handling Project M3 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.

#1. AI model selection



The cleaned dataset was stored in Google Cloud Storage and accessed directly from the bucket using a gs:// URI. This approach supports reproducibility and demonstrates integration with cloud-based data infrastructure.

The file AI model selection.ipynb presents the code for the AI selection:

https://github.com/maryslopes/AI-Data-Handling/blob/M3-project/AI%20model%20selection.ipynb

Six original features were selected for machine failure prediction. The categorical machine type variable was transformed using one-hot encoding, producing two binary variables (Type_L and Type_M). This resulted in seven preprocessed features used by the SVM and Neural Network models.

An SVM classifier using an RBF kernel was implemented to capture non-linear relationships between equipment sensor measurements and machine failures. A regularisation parameter C=1.0 and gamma='scale' were used as baseline settings. A fixed random state of 42 was applied to ensure reproducibility.

The RBF kernel was selected because equipment failures are influenced by complex non-linear relationships between sensor measurements.

The original dataset contained 15 variables. Feature selection was performed to retain only the attributes considered relevant for predicting machine failure: machine type, air temperature, process temperature, rotational speed, torque and tool wear. The categorical machine type variable was one-hot encoded, resulting in a final feature set of seven input variables used for model training.

a. Preprocessing strategie

Three preprocessing strategies were evaluated: unscaled features, standardisation (StandardScaler), and MinMax normalisation. The unscaled data resulted in poor neural network convergence, with training loss remaining high throughout training. Standardisation improved optimisation by transforming features to zero mean and unit variance, leading to a substantial reduction in training loss. MinMax normalisation scaled features to the range [0,1] and produced the highest classification accuracy (97.3%) with the neural network and 96.7% with the SVM. These results demonstrate that feature scaling is critical for machine-failure prediction models.

Two classification models were evaluated for machine failure prediction: a Support Vector Machine (SVM) and a feedforward Neural Network. The SVM achieved 96.7% accuracy while requiring only 0.17 seconds of training time. The Neural Network achieved the highest accuracy (97.3%) but required substantially more computational time (27.9 seconds). These results demonstrate a trade-off between predictive performance and computational efficiency. While the Neural Network produced the best classification accuracy, the SVM offered comparable performance with significantly lower training costs.
As seen below: 
Method	SVM Accuracy	NN Accuracy	SVM Time (s)	NN (Time (s)
Unscaled	0.0340	0.9660	0.5906	33.4541
StandardScaler	0.9180	0.9750	0.8717	27.4825
MinMaxScaler	0.9670	0.9730	0.1730	27.8635

<img width="657" height="173" alt="image" src="https://github.com/user-attachments/assets/4b0123b3-31d5-47af-b7e4-2aa8a6f7c440" />


The GPU achieved a 0.3% higher accuracy than CPU however it required 0.15s longer. They are substantially comparing the ratio accurancy/time. Therefore is not worthy to use GPU for this model.
Method	NN Accuracy	NN (Time (s)
StandardScaler CPU	0.9750	27.4825
StandardScaler GPU	0.9780	27.6330

<img width="440" height="88" alt="image" src="https://github.com/user-attachments/assets/14105e61-5c29-4beb-a289-24241dbd0691" />


b. Learning Curve Plot
The learning curves demonstrate that feature scaling substantially improved neural network training. Models trained on standardised data showed a continuous reduction in training loss, whereas models trained on unscaled features exhibited persistently high loss values. This suggests that standardisation improved optimisation and convergence.

<img width="850" height="552" alt="image" src="https://github.com/user-attachments/assets/61ff1d91-4fd3-4bbb-96e2-cb9ed83b2d88" />

c. Misclassified Samples vs Feature Distributions
he largest spread of error points appears around Torque [Nm] and Tool wear [min]. hese variables may contribute more to prediction uncertainty.

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/a849c9b6-3b39-4788-89a0-35c91b144ac1" />

#2.Hyperparameter optimization

The Hyperparameter.ipynb code presents the analysis. 

The best Hyperparameter Set Found was:

<img width="697" height="87" alt="image" src="https://github.com/user-attachments/assets/731b63d0-fc1a-4500-b001-c8f4796bf060" />


#3.MLOps and experiment tracking

The search and retrieve the most recent MLflow run resulted on: 

<img width="907" height="452" alt="image" src="https://github.com/user-attachments/assets/a46bedb3-5127-48b8-95e3-c74f4e3ebc3c" />

#4.Analysis and conclusions from experiments


#5.Compute architecture selection

 
#6.Reproducibility 

The project environment was documented using a concise requirements.txt file containing the exact package versions required to reproduce the experiments.

#7.Model card / model documentation


#8. Error analysis and model weaknesses



