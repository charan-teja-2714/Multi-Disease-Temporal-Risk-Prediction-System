import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';
// const API_BASE_URL = 'https://multi-disease-temporal-risk-prediction.onrender.com';
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Automatically attach the current user's ID to every request
api.interceptors.request.use((config) => {
  try {
    const stored = sessionStorage.getItem('user');
    if (stored) {
      const user = JSON.parse(stored);
      if (user?.id) config.headers['X-User-Id'] = user.id;
    }
  } catch {}
  return config;
});

// Patient API calls
export const patientAPI = {
  // Get all patients
  getPatients: async () => {
    const response = await api.get('/patients/');
    return response.data;
  },

  // Get specific patient
  getPatient: async (patientId) => {
    const response = await api.get(`/patients/${patientId}`);
    return response.data;
  },

  // Create new patient
  createPatient: async (patientData) => {
    const response = await api.post('/patients/', patientData);
    return response.data;
  },

  // Update patient
  updatePatient: async (patientId, patientData) => {
    const response = await api.put(`/patients/${patientId}`, patientData);
    return response.data;
  },

  // Delete patient
  deletePatient: async (patientId) => {
    const response = await api.delete(`/patients/${patientId}`);
    return response.data;
  },

  // Get patient health records
  getHealthRecords: async (patientId) => {
    const response = await api.get(`/health-records/${patientId}`);
    return response.data;
  },

  // Add health record
  addHealthRecord: async (recordData) => {
    const response = await api.post('/health-records/', recordData);
    return response.data;
  },

  // Delete a single health record
  deleteHealthRecord: async (recordId) => {
    const response = await api.delete(`/health-records/record/${recordId}`);
    return response.data;
  },

  // Get patient predictions
  getPredictions: async (patientId) => {
    const response = await api.get(`/predictions/${patientId}`);
    return response.data;
  },

  // Generate new prediction
  generatePrediction: async (patientId) => {
    const response = await api.post(`/predict/${patientId}`);
    return response.data;
  },

  // Upload medical report
  uploadMedicalReport: async (patientId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post(`/extract-report/${patientId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }
};

// Auth API calls
export const authAPI = {
  register: async (username, email, password) => {
    const response = await api.post('/auth/register', { username, email, password });
    return response.data;
  },

  login: async (username_or_email, password) => {
    const response = await api.post('/auth/login', { username_or_email, password });
    return response.data;
  }
};

// System API calls
export const systemAPI = {
  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Get model info
  getModelInfo: async () => {
    const response = await api.get('/model-info');
    return response.data;
  }
};

export default api;