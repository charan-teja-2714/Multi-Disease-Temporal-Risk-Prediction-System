# Multi-Disease Temporal Risk Prediction System
## Viva Answers - Part 1 (Questions 1-90)

---

## **1. PROJECT OVERVIEW & MOTIVATION**

### Basic Understanding

**1. What is the main objective of your project?**
To predict future risks of diabetes, heart disease, and kidney disease by analyzing longitudinal patient health records using temporal deep learning models (TCN and Transformer).

**2. Why did you choose to predict multiple diseases instead of just one?**
Because these diseases are medically related (diabetes often leads to kidney disease, heart disease shares risk factors with diabetes). Multi-task learning allows the model to learn shared patterns, improving accuracy for all three diseases.

**3. What are the three diseases your system predicts?**
- Diabetes Mellitus
- Cardiovascular/Heart Disease
- Chronic Kidney Disease (CKD)

**4. What is temporal risk prediction? How is it different from regular prediction?**
Temporal prediction analyzes health data over time (multiple visits) to identify disease progression patterns. Regular prediction uses a single snapshot. Temporal prediction captures trends like "glucose increasing over 6 months" which is more predictive than a single glucose reading.

**5. Who are the end users of this system?**
Primary care physicians, endocrinologists, cardiologists, and nephrologists who monitor patients with chronic disease risk factors.

### Problem Statement

**6. What problem does your system solve in healthcare?**
Early detection of disease risk before symptoms appear, allowing preventive interventions. It helps doctors identify high-risk patients who need closer monitoring or lifestyle changes.

**7. Why is longitudinal data important for disease prediction?**
Diseases develop gradually over months/years. A single abnormal reading might be temporary, but consistent trends (e.g., rising HbA1c over 3 visits) indicate true disease progression.

**8. What challenges exist in analyzing patient health records over time?**
- Irregular visit intervals (patients don't visit at fixed times)
- Missing values (not all tests done at every visit)
- Variable sequence lengths (different patients have different visit counts)
- Temporal dependencies (current health depends on past health)

**9. How does your system handle irregular visit intervals?**
By encoding time gaps as features (days_since_first_visit, days_since_last_visit) and using models (TCN, Transformer) that can process variable-length sequences with temporal encoding.

**10. What is the clinical significance of predicting disease risk in advance?**
Enables preventive care, reduces healthcare costs, improves patient outcomes, and allows early lifestyle interventions before irreversible damage occurs.

---

## **2. DATASET & DATA PROCESSING**

### Data Source

**11. What dataset did you use for this project?**
Synthetic healthcare data generated to simulate realistic longitudinal patient records with disease progression patterns.

**12. Why did you use synthetic data instead of real patient data?**
- Real datasets like MIMIC-IV require special access and IRB approval
- Privacy concerns with real patient data
- Synthetic data allows controlled experiments
- Can generate specific patterns for demonstration

**13. How did you generate synthetic healthcare data?**
Using SyntheticHealthDataGenerator class that:
- Creates patient profiles with risk factors
- Generates baseline health values
- Simulates disease progression over time with trends
- Adds realistic noise and missing values

**14. What are the key health metrics/features in your dataset?**
11 health features: glucose, HbA1c, creatinine, BUN, systolic BP, diastolic BP, cholesterol, HDL, LDL, triglycerides, BMI
Plus 2 temporal features: days_since_first_visit, days_since_last_visit

**15. How many patients and health records did you generate?**
500-1000 patients with 3-15 visits each, resulting in ~5000-10000 health records spanning 1-5 years per patient.

### Data Characteristics

**16. What is the time span of patient records in your dataset?**
12 to 60 months (1-5 years) per patient, with irregular visit intervals of 1-6 months between visits.

**17. How did you simulate disease progression over time?**
Applied time-dependent trends to health metrics based on risk factors. For example, if diabetes_risk > 0.7, glucose increases by 30% over the time period with added noise.

**18. What are normal ranges for glucose, HbA1c, creatinine, etc.?**
- Glucose: 70-100 mg/dL (fasting)
- HbA1c: 4.0-5.6%
- Creatinine: 0.6-1.2 mg/dL
- BUN: 7-20 mg/dL
- Systolic BP: 90-120 mmHg
- Cholesterol: 125-200 mg/dL

**19. Why did you include missing values in synthetic data?**
To simulate real clinical scenarios where not all tests are performed at every visit (10% missing rate reflects reality).

**20. How do irregular visit intervals reflect real clinical scenarios?**
Patients don't visit at fixed intervals - they come when sick, for routine checkups, or based on doctor recommendations. This creates realistic temporal patterns.

### Preprocessing

**21. How do you handle missing values in health records?**
Three-step approach:
1. Forward fill within patient (carry last observation forward)
2. Backward fill for remaining gaps
3. Fill any remaining with median value across all patients

**22. What is forward fill and backward fill? Why use them?**
- Forward fill: Use previous visit's value (assumes health status persists)
- Backward fill: Use next visit's value (for gaps at start)
- Preserves temporal continuity better than mean imputation

**23. Why do you normalize/scale the features?**
Different features have different scales (glucose: 70-200, creatinine: 0.6-2.0). Neural networks train better when all inputs are on similar scales (mean=0, std=1).

**24. What is StandardScaler and how does it work?**
Transforms features to have mean=0 and standard deviation=1 using: z = (x - μ) / σ
Fitted on training data, then applied to validation/test to prevent data leakage.

**25. How do you create fixed-length sequences from variable-length patient histories?**
- If patient has ≥10 visits: take last 10 visits
- If patient has <10 visits: pad beginning with zeros
- This creates uniform (batch_size, 10, 13) tensors

**26. What is sequence padding and why is it necessary?**
Adding zeros to short sequences so all sequences have the same length. Neural networks require fixed input dimensions for batch processing.

**27. How do you encode temporal information (time gaps between visits)?**
Two features:
- days_since_first_visit: cumulative time from baseline
- days_since_last_visit: interval between consecutive visits
These capture both absolute time and visit frequency.

**28. What are the 13 input features to your model?**
11 health metrics + 2 temporal features = 13 total features per time step.

---

## **3. DEEP LEARNING MODELS**

### Model Selection

**29. Why did you NOT use LSTM or GRU models?**
Project requirement to explore modern alternatives. Also:
- LSTMs have vanishing gradient issues with long sequences
- TCN and Transformers are faster (parallelizable)
- Recent research shows TCN/Transformers outperform RNNs on time-series tasks

**30. What are the two models you implemented?**
1. Temporal Convolutional Network (TCN)
2. Time-Series Transformer

**31. What is a Temporal Convolutional Network (TCN)?**
A CNN-based architecture using dilated causal convolutions to process sequences. Unlike standard CNNs, it maintains temporal order and has exponentially growing receptive fields.

**32. What is a Time-Series Transformer?**
Transformer architecture adapted for time-series with positional encoding and time-gap encoding to handle irregular temporal data.

**33. Why did you implement two different models?**
To compare approaches: TCN (convolutional) vs Transformer (attention-based). Provides insights into which architecture works better for medical time-series.

### Temporal Convolutional Network (TCN)

**34. What are dilated convolutions?**
Convolutions with gaps between kernel elements. Dilation rate d means kernel elements are d positions apart, allowing larger receptive fields without more parameters.

**35. What is causal convolution and why is it important?**
Convolution that only uses past and current information, never future. Critical for time-series to prevent data leakage (model can't "see the future").

**36. How does dilation help capture long-term dependencies?**
With dilation rates [1, 2, 4, 8], a 3-layer network sees 1+2+4+8 = 15 time steps back with only 3 layers. Exponential growth in receptive field.

**37. What is the receptive field in TCN?**
The number of past time steps that influence the current output. With kernel size k and L layers with dilations [1,2,4,...,2^(L-1)], receptive field = 1 + 2(k-1)∑2^i.

**38. What are residual connections and why use them?**
Skip connections that add input directly to output: output = F(x) + x. Helps gradient flow during backpropagation, enabling deeper networks without vanishing gradients.

**39. How many TCN layers did you use and why?**
3 layers with channels [64, 64, 64] or [32, 32, 32]. Balances model capacity with training speed for demonstration purposes.

**40. What are the advantages of TCN over RNNs?**
- Parallelizable (faster training)
- Stable gradients (no vanishing/exploding)
- Flexible receptive field (controlled by dilation)
- Lower memory usage
- Better long-term dependencies

**41. How does TCN avoid vanishing gradient problem?**
Residual connections provide direct gradient paths, and dilated convolutions don't have the sequential dependency that causes gradient decay in RNNs.

### Time-Series Transformer

**42. What is self-attention mechanism?**
Mechanism that computes weighted relationships between all positions in a sequence. Each position "attends to" all other positions to determine its representation.

**43. How does multi-head attention work?**
Runs multiple attention mechanisms in parallel (8 heads in our case), each learning different types of relationships. Outputs are concatenated and projected.

**44. What is positional encoding in Transformers?**
Since Transformers have no inherent notion of order, positional encoding adds position information using sine/cosine functions: PE(pos,2i) = sin(pos/10000^(2i/d)).

**45. How do you handle irregular time gaps in Transformer?**
Added TimeGapEncoding layer that embeds the actual time difference (in days) between visits, allowing the model to learn that a 30-day gap differs from a 180-day gap.

**46. What is the difference between encoder and decoder in Transformer?**
Encoder processes input sequence, decoder generates output sequence. For our classification task, we only use the encoder part.

**47. Why is Transformer suitable for time-series data?**
Attention mechanism can focus on relevant past time points regardless of distance, capturing long-range dependencies better than fixed-window approaches.

**48. How many attention heads did you use?**
8 attention heads in the full model, 4 in the smaller demo version.

**49. What is the model dimension (d_model) in your Transformer?**
128 in full model, 64 in demo version. This is the size of the hidden representations.

### Multi-Task Learning

**50. What is multi-task learning?**
Training a single model to perform multiple related tasks simultaneously by sharing representations while having task-specific output layers.

**51. Why predict multiple diseases together instead of separately?**
- Diseases share risk factors (obesity affects all three)
- Shared patterns improve learning efficiency
- Reduces total parameters vs 3 separate models
- Medically realistic (comorbidities are common)

**52. How are the three diseases related medically?**
- Diabetes damages blood vessels → heart disease
- Diabetes damages kidney filters → kidney disease
- High blood pressure affects both heart and kidneys
- All share metabolic risk factors

**53. What is a shared encoder in multi-task learning?**
The TCN or Transformer layers that process input sequences and learn common temporal patterns across all diseases.

**54. What are disease-specific heads?**
Separate small neural networks (2-3 layers) for each disease that take shared features and produce disease-specific risk predictions.

**55. How does multi-task learning improve accuracy?**
Shared representations learn general health patterns, reducing overfitting. Gradients from all tasks provide richer learning signal.

**56. What is the architecture of your multi-task model?**
```
Input (10×13) → Shared Encoder (TCN/Transformer) → Shared FC (128→64) 
→ ├─ Diabetes Head → Risk
  ├─ Heart Head → Risk  
  └─ Kidney Head → Risk
```

---

## **4. TRAINING & OPTIMIZATION**

### Training Process

**57. What loss function did you use?**
Binary Cross-Entropy (BCE) loss for each disease, combined into a weighted multi-task loss.

**58. What is Binary Cross-Entropy (BCE) loss?**
Loss for binary classification: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
Measures difference between predicted probability and true label.

**59. How do you combine losses from three diseases?**
Weighted sum: Total_Loss = w₁·L_diabetes + w₂·L_heart + w₃·L_kidney
Default weights are [1.0, 1.0, 1.0] (equal importance).

**60. What optimizer did you use and why?**
Adam optimizer - combines momentum and adaptive learning rates, works well for most deep learning tasks without extensive tuning.

**61. What is the learning rate you used?**
0.001 (1e-3) - standard default for Adam optimizer.

**62. How many epochs did you train for?**
Up to 50 epochs with early stopping (typically stops around 20-30 epochs).

**63. What is early stopping and why use it?**
Stops training when validation loss doesn't improve for N epochs (patience). Prevents overfitting and saves training time.

**64. What is the patience parameter in early stopping?**
Number of epochs to wait for improvement before stopping. We use patience=10, meaning if validation loss doesn't improve for 10 consecutive epochs, training stops.

### Hyperparameters

**65. What batch size did you use?**
32 - balances memory usage and gradient stability.

**66. How did you split data into train/validation/test sets?**
70% training, 15% validation, 15% test (randomly shuffled).

**67. What is the train-validation-test ratio?**
70:15:15 split.

**68. Why do you need a validation set?**
To tune hyperparameters and monitor overfitting without touching the test set. Test set is only used for final evaluation.

**69. What is learning rate scheduling?**
Dynamically adjusting learning rate during training. We use ReduceLROnPlateau which reduces LR when validation loss plateaus.

**70. What is ReduceLROnPlateau scheduler?**
Reduces learning rate by a factor (0.5) when validation loss doesn't improve for N epochs (patience=5). Helps fine-tune in later epochs.

### Evaluation Metrics

**71. What metrics did you use to evaluate your model?**
- AUC-ROC score
- Precision, Recall, F1-score
- Training/Validation loss curves

**72. What is AUC-ROC score?**
Area Under the Receiver Operating Characteristic curve. Measures model's ability to distinguish between classes across all thresholds. Range: 0-1, where 0.5 = random, 1.0 = perfect.

**73. What is precision, recall, and F1-score?**
- Precision: TP/(TP+FP) - of predicted positives, how many are correct
- Recall: TP/(TP+FN) - of actual positives, how many we found
- F1: 2·(P·R)/(P+R) - harmonic mean of precision and recall

**74. Why is AUC important for medical predictions?**
Threshold-independent metric, handles class imbalance well, and directly measures discrimination ability - critical when false negatives (missing disease) are costly.

**75. What AUC scores did your models achieve?**
Varies by training run, typically 0.65-0.85 on synthetic data. Note: untrained model in API shows ~0.5 (random).

**76. How do you interpret a 0.5 AUC score?**
Model performs no better than random guessing - indicates model hasn't learned meaningful patterns (often due to insufficient training or poor data).

**77. What is the difference between training loss and validation loss?**
- Training loss: measured on data model sees during training
- Validation loss: measured on held-out data
- Gap between them indicates overfitting (model memorizing training data)

---

## **5. EXPLAINABILITY (SHAP)**

### Concept

**78. What is explainable AI (XAI)?**
Techniques to make AI model decisions interpretable and understandable to humans, showing which inputs influenced which outputs.

**79. Why is explainability important in healthcare?**
- Doctors need to trust and verify AI recommendations
- Regulatory requirements (FDA, GDPR)
- Ethical responsibility for medical decisions
- Helps identify model biases or errors
- Educational value for medical training

**80. What is SHAP?**
SHapley Additive exPlanations - a unified framework for interpreting model predictions based on game theory.

**81. What does SHAP stand for?**
SHapley Additive exPlanations (based on Shapley values from cooperative game theory).

**82. How does SHAP work?**
Computes contribution of each feature by comparing predictions with and without that feature, averaged over all possible feature combinations. Satisfies desirable properties: local accuracy, missingness, consistency.

**83. What are SHAP values?**
Numerical values indicating each feature's contribution to pushing the prediction away from the base value. Positive = increases risk, negative = decreases risk.

### Implementation

**84. How did you integrate SHAP into your system?**
Created MedicalExplainer class that wraps the model, computes SHAP values for predictions, and generates human-readable explanations.

**85. What is feature importance?**
Ranking of features by their average absolute SHAP values - shows which health metrics matter most for predictions.

**86. What is temporal importance?**
Importance of each time step (visit) in the sequence - shows which visits were most predictive.

**87. How do you generate doctor-friendly explanations?**
Convert SHAP values to natural language: "Blood sugar levels (145.2 mg/dL) increase diabetes risk. Your HbA1c (7.8%) indicates poor long-term glucose control."

**88. Which features are most important for diabetes prediction?**
Glucose, HbA1c (primary markers), BMI, and temporal trends in these values.

**89. Which features are most important for heart disease?**
Systolic BP, cholesterol, LDL, HDL, and BMI.

**90. Which features are most important for kidney disease?**
Creatinine, BUN (direct kidney function markers), plus diabetes-related features (diabetes causes kidney disease).

---

**Continue to Part 2 for questions 91-180...**
