# Multi-Disease Temporal Risk Prediction System
## Viva Answers - Part 2 (Questions 91-180)

---

## **6. SYSTEM ARCHITECTURE**

### Backend

**91. What framework did you use for the backend?**
FastAPI - a modern Python web framework for building APIs.

**92. Why did you choose FastAPI?**
- Fast performance (based on Starlette and Pydantic)
- Automatic API documentation (Swagger UI)
- Type hints and validation
- Async support
- Easy integration with ML models

**93. What is REST API?**
Representational State Transfer - architectural style using HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources.

**94. What database did you use?**
SQLite - a lightweight, file-based relational database.

**95. Why SQLite instead of PostgreSQL or MySQL?**
- No separate server needed (embedded database)
- Perfect for development and demonstration
- Single file storage (easy to share/backup)
- Sufficient for small-scale deployment
- Would upgrade to PostgreSQL for production

**96. What are the main API endpoints?**
- POST /patients/ - Create patient
- GET /patients/ - List patients
- POST /health-records/ - Add health record
- GET /health-records/{id} - Get patient records
- POST /predict/{id} - Generate prediction
- GET /predictions/{id} - Get prediction history

**97. How do you handle CORS in your API?**
Using CORSMiddleware to allow requests from frontend (localhost:3000), enabling cross-origin requests between frontend and backend.

### Database Schema

**98. What tables are in your database?**
Three tables:
- patients (id, name, age, gender, created_at)
- health_records (id, patient_id, visit_date, glucose, hba1c, ...)
- predictions (id, patient_id, prediction_date, diabetes_risk, heart_disease_risk, kidney_disease_risk, explanation)

**99. What is the relationship between patients and health records?**
One-to-many: One patient has many health records. Foreign key patient_id in health_records references patients.id.

**100. How do you store predictions in the database?**
Each prediction is a row with patient_id, timestamp, three risk scores (0-1), and explanation text.

**101. What is SQLAlchemy?**
Python SQL toolkit and ORM (Object-Relational Mapping) that lets you interact with databases using Python objects instead of raw SQL.

**102. What is an ORM (Object-Relational Mapping)?**
Technique to map database tables to Python classes, rows to objects, and columns to attributes. Allows database operations using object-oriented code.

### Frontend

**103. What framework did you use for the frontend?**
React.js - a JavaScript library for building user interfaces.

**104. Why React.js?**
- Component-based architecture (reusable UI elements)
- Virtual DOM for performance
- Large ecosystem and community
- Easy state management
- Industry standard for modern web apps

**105. What UI library did you use (Ant Design)?**
Ant Design - professional React UI component library with medical-appropriate styling.

**106. What are the main pages in your application?**
- Dashboard - System overview and statistics
- PatientList - Browse and add patients
- PatientDetail - View patient history and predictions
- AddHealthRecord - Enter health measurements

**107. How does the frontend communicate with the backend?**
HTTP requests using Axios library to call REST API endpoints (GET, POST).

**108. What is Axios?**
Promise-based HTTP client for JavaScript that simplifies making API requests from the browser.

---

## **7. TECHNICAL IMPLEMENTATION**

### PyTorch

**109. Why did you use PyTorch instead of TensorFlow?**
- More Pythonic and intuitive API
- Dynamic computation graphs (easier debugging)
- Strong research community
- Better for custom architectures
- Personal preference and familiarity

**110. What is a PyTorch Dataset?**
Abstract class that represents a dataset. Must implement __len__() and __getitem__() methods to enable indexing and iteration.

**111. What is a DataLoader?**
Utility that wraps a Dataset and provides batching, shuffling, and parallel data loading for training.

**112. What is the purpose of model.eval()?**
Switches model to evaluation mode - disables dropout and batch normalization training behavior. Critical for inference.

**113. What is torch.no_grad()?**
Context manager that disables gradient computation, reducing memory usage and speeding up inference when you don't need backpropagation.

**114. How do you save and load PyTorch models?**
```python
# Save
torch.save(model.state_dict(), 'model.pth')
# Load
model.load_state_dict(torch.load('model.pth'))
```

### Model Architecture Details

**115. How many parameters does your TCN model have?**
Approximately 50,000-100,000 parameters depending on configuration (channels=[32,32,32] vs [64,64,64]).

**116. How many parameters does your Transformer model have?**
Approximately 200,000-500,000 parameters (larger due to attention mechanisms).

**117. What activation functions did you use?**
ReLU (Rectified Linear Unit) - f(x) = max(0, x). Simple, effective, no vanishing gradient for positive values.

**118. What is dropout and why use it?**
Regularization technique that randomly sets a fraction of neurons to zero during training. Prevents overfitting by forcing network to learn redundant representations.

**119. What dropout rate did you use?**
0.2 (20%) - drops 20% of neurons during training.

**120. What is batch normalization?**
Normalizes layer inputs across the batch dimension. Stabilizes training, allows higher learning rates, and acts as regularization.

### Sequence Processing

**121. What is the sequence length you used (10 visits)?**
10 time steps - each representing one patient visit.

**122. Why did you choose 10 as the sequence length?**
Balance between:
- Capturing sufficient history (10 visits ≈ 2-3 years)
- Computational efficiency
- Most patients have at least 3-5 visits
- Manageable memory usage

**123. How do you handle patients with fewer than 10 visits?**
Zero-padding at the beginning of the sequence. Model learns to ignore padded positions.

**124. How do you handle patients with more than 10 visits?**
Take the most recent 10 visits (last 10 in chronological order).

**125. What happens if a patient has only 1 visit?**
System requires minimum 2 visits for prediction (cannot establish temporal trends with single point).

---

## **8. MEDICAL DOMAIN KNOWLEDGE**

### Diabetes

**126. What is diabetes mellitus?**
Chronic metabolic disorder characterized by high blood glucose due to insufficient insulin production (Type 1) or insulin resistance (Type 2).

**127. What is glucose and what are normal levels?**
Blood sugar - primary energy source. Normal fasting: 70-100 mg/dL. Prediabetes: 100-125. Diabetes: ≥126 mg/dL.

**128. What is HbA1c and why is it important?**
Glycated hemoglobin - measures average blood glucose over 2-3 months. More reliable than single glucose reading. Normal: <5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%.

**129. What HbA1c level indicates diabetes?**
≥6.5% on two separate tests indicates diabetes diagnosis.

**130. What are risk factors for diabetes?**
Obesity, family history, sedentary lifestyle, age >45, high blood pressure, abnormal cholesterol, history of gestational diabetes.

### Heart Disease

**131. What is cardiovascular disease?**
Diseases affecting heart and blood vessels, including coronary artery disease, heart failure, arrhythmias, and stroke.

**132. What is systolic and diastolic blood pressure?**
- Systolic: pressure when heart beats (pumping). Normal: <120 mmHg
- Diastolic: pressure when heart rests between beats. Normal: <80 mmHg

**133. What blood pressure is considered hypertension?**
≥140/90 mmHg (Stage 1: 130-139/80-89, Stage 2: ≥140/90).

**134. What is cholesterol (HDL, LDL, triglycerides)?**
- Total cholesterol: all cholesterol in blood. Desirable: <200 mg/dL
- LDL (bad): deposits in arteries. Optimal: <100 mg/dL
- HDL (good): removes cholesterol. Desirable: >60 mg/dL
- Triglycerides: blood fats. Normal: <150 mg/dL

**135. What cholesterol levels are considered high?**
Total cholesterol >240 mg/dL, LDL >160 mg/dL, Triglycerides >200 mg/dL.

### Kidney Disease

**136. What is chronic kidney disease (CKD)?**
Progressive loss of kidney function over months/years. Kidneys filter waste from blood; CKD leads to waste buildup.

**137. What is creatinine and why is it important?**
Waste product from muscle metabolism. Kidneys filter it out. High creatinine indicates poor kidney function. Normal: 0.6-1.2 mg/dL.

**138. What is BUN (Blood Urea Nitrogen)?**
Waste product from protein breakdown. Elevated BUN indicates kidney dysfunction or dehydration. Normal: 7-20 mg/dL.

**139. What creatinine level indicates kidney dysfunction?**
>1.5 mg/dL suggests impaired kidney function. >2.0 indicates significant dysfunction.

**140. How are diabetes and kidney disease related?**
Diabetes is leading cause of CKD. High glucose damages kidney blood vessels and filters (glomeruli), leading to diabetic nephropathy.

### Disease Relationships

**141. How does diabetes lead to kidney disease?**
Chronic high glucose damages glomeruli (kidney filters), causing protein leakage and progressive kidney failure. ~40% of diabetics develop kidney disease.

**142. How are heart disease and diabetes related?**
Diabetes damages blood vessels, increases atherosclerosis risk, and is a major risk factor for heart attack and stroke. Diabetics have 2-4x higher heart disease risk.

**143. What is metabolic syndrome?**
Cluster of conditions (high BP, high glucose, excess abdominal fat, abnormal cholesterol) that increase risk of heart disease, stroke, and diabetes.

**144. Why predict multiple diseases together?**
They share common pathophysiology and risk factors. Predicting together captures these relationships and provides comprehensive patient risk assessment.

---

## **9. SYSTEM WORKFLOW**

### User Workflow

**145. How does a doctor use your system?**
1. Login to dashboard
2. Search/select patient or create new patient
3. View patient history
4. Add new health records from recent visit
5. Click "Generate Prediction"
6. Review risk scores and explanations
7. Make clinical decisions based on AI insights

**146. What is the minimum data required for prediction?**
At least 2 health records (visits) for the patient to establish temporal trends.

**147. How do you add a new patient?**
Navigate to Patients page → Click "Add New Patient" → Enter name, age, gender → Submit.

**148. How do you add health records?**
Navigate to "Add Health Record" → Select patient → Enter visit date → Fill in available health metrics → Submit.

**149. How do you generate a prediction?**
Go to patient detail page → Click "Generate Prediction" button → System processes health history → Displays risk scores.

**150. How are predictions displayed to the user?**
Circular progress bars showing risk percentage (0-100%) with color coding:
- Green: Low risk (<30%)
- Yellow: Moderate risk (30-70%)
- Red: High risk (>70%)
Plus textual explanation of contributing factors.

### Data Flow

**151. Explain the complete data flow from input to prediction.**
```
User Input → Frontend Form → HTTP POST → FastAPI Endpoint 
→ Database Query (get health records) → Tensor Creation 
→ Model Inference → Risk Scores → Database Save 
→ HTTP Response → Frontend Display
```

**152. How is data stored in the database?**
Using SQLAlchemy ORM: Python objects → SQL INSERT statements → SQLite file (medical_predictions.db).

**153. How is data retrieved for prediction?**
SQL query: SELECT * FROM health_records WHERE patient_id=X ORDER BY visit_date → Convert to Python objects → Extract features.

**154. How are predictions saved?**
Create Prediction object with patient_id, risks, explanation → SQLAlchemy session.add() → session.commit() → Saved to database.

**155. Can you view prediction history?**
Yes, patient detail page shows all past predictions with timestamps in reverse chronological order.

---

## **10. CHALLENGES & SOLUTIONS**

### Technical Challenges

**156. What was the biggest technical challenge you faced?**
Integrating the complex preprocessing pipeline with real-time API predictions while handling missing values and irregular time intervals.

**157. How did you handle missing values in real-time predictions?**
Simplified approach: replace None with 0.0 during tensor creation. More sophisticated preprocessing is done during training.

**158. Why did you simplify the prediction pipeline?**
The full preprocessor (with pandas operations) caused errors in production. Simplified version creates tensors directly from database records, trading some sophistication for reliability.

**159. What pandas deprecation warnings did you encounter?**
- fillna(method='ffill') → ffill()
- fillna(method='bfill') → bfill()
- groupby().apply() without include_groups parameter
These are warnings about future pandas versions, not errors.

**160. How did you fix the "mix of continuous and binary targets" error?**
Converted continuous target labels (0.0-1.0 risk scores) to binary (0 or 1) using 0.5 threshold before calculating precision/recall metrics.

### Model Challenges

**161. Why did your model show 0.5 AUC during training?**
Possible causes:
- Insufficient training epochs
- Poor data quality or labeling
- Model not learning (check loss decreasing)
- Class imbalance
- Random initialization not converged

**162. What is overfitting and how did you prevent it?**
Model memorizes training data instead of learning patterns. Prevention:
- Dropout (0.2)
- Early stopping
- Train/validation split
- Regularization through multi-task learning

**163. What is the vanishing gradient problem?**
In deep networks, gradients become extremely small during backpropagation, preventing early layers from learning. Common in RNNs with long sequences.

**164. How does TCN avoid vanishing gradients?**
Residual connections provide direct gradient paths, and dilated convolutions don't have the sequential bottleneck of RNNs.

---

**Continue to Part 3 for questions 181-270...**
