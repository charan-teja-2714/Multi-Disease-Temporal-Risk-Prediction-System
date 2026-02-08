import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Progress,
  Alert,
  Spin,
  Typography,
  Tag,
  Space,
  message,
  Divider,
  Upload,
  Modal
} from 'antd';
import {
  UserOutlined,
  HeartOutlined,
  ExperimentOutlined,
  CalendarOutlined,
  WarningOutlined,
  UploadOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { patientAPI } from '../services/api';
import moment from 'moment';

const { Title, Text, Paragraph } = Typography;

const PatientDetail = () => {
  const { id } = useParams();
  const [patient, setPatient] = useState(null);
  const [healthRecords, setHealthRecords] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);

  const loadPatientData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Load patient info
      const patientData = await patientAPI.getPatient(id);
      setPatient(patientData);

      // Load health records
      const recordsData = await patientAPI.getHealthRecords(id);
      setHealthRecords(recordsData);

      // Load predictions
      const predictionsData = await patientAPI.getPredictions(id);
      setPredictions(predictionsData);

    } catch (error) {
      message.error('Failed to load patient data');
      console.error('Error loading patient data:', error);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadPatientData();
  }, [loadPatientData]);

  const generatePrediction = async () => {
    try {
      setPredicting(true);
      await patientAPI.generatePrediction(id);
      message.success('Prediction generated successfully');
      loadPatientData(); // Reload to show new prediction
    } catch (error) {
      message.error('Failed to generate prediction. Ensure patient has at least 1 health record.');
      console.error('Error generating prediction:', error);
    } finally {
      setPredicting(false);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setUploading(true);
      const result = await patientAPI.uploadMedicalReport(id, file);
      
      if (result.success) {
        message.success(
          `Successfully processed report! Extracted ${result.extracted_fields} fields, merged ${result.merged_fields} new values.`
        );
        loadPatientData(); // Reload to show new health records
        setUploadModalVisible(false);
      } else {
        message.warning(result.message || 'No new data extracted from report');
      }
    } catch (error) {
      message.error('Failed to process medical report');
      console.error('Error uploading report:', error);
    } finally {
      setUploading(false);
    }
    
    return false; // Prevent default upload behavior
  };

  const getRiskColor = (risk) => {
    if (risk < 0.3) return '#52c41a'; // Green
    if (risk < 0.7) return '#faad14'; // Orange
    return '#ff4d4f'; // Red
  };

  const getRiskLevel = (risk) => {
    if (risk < 0.3) return 'Low';
    if (risk < 0.7) return 'Moderate';
    return 'High';
  };

  // Prepare chart data
  const chartData = healthRecords.map((record, index) => ({
    visit: `Visit ${index + 1}`,
    date: moment(record.visit_date).format('MM/DD'),
    glucose: record.glucose,
    systolic_bp: record.systolic_bp,
    creatinine: record.creatinine,
    cholesterol: record.cholesterol
  }));

  const healthRecordColumns = [
    {
      title: 'Date',
      dataIndex: 'visit_date',
      key: 'visit_date',
      render: (date) => moment(date).format('YYYY-MM-DD'),
    },
    {
      title: 'Source',
      dataIndex: 'source',
      key: 'source',
      render: (source) => (
        <Tag color={source === 'rag_report' ? 'blue' : 'green'}>
          {source === 'rag_report' ? 'Report' : 'Manual'}
        </Tag>
      ),
    },
    {
      title: 'Glucose',
      dataIndex: 'glucose',
      key: 'glucose',
      render: (value) => value ? `${value.toFixed(1)} mg/dL` : 'N/A',
    },
    {
      title: 'HbA1c',
      dataIndex: 'hba1c',
      key: 'hba1c',
      render: (value) => value ? `${value.toFixed(1)}%` : 'N/A',
    },
    {
      title: 'BP (Systolic)',
      dataIndex: 'systolic_bp',
      key: 'systolic_bp',
      render: (value) => value ? `${value.toFixed(0)} mmHg` : 'N/A',
    },
    {
      title: 'Creatinine',
      dataIndex: 'creatinine',
      key: 'creatinine',
      render: (value) => value ? `${value.toFixed(2)} mg/dL` : 'N/A',
    },
    {
      title: 'Cholesterol',
      dataIndex: 'cholesterol',
      key: 'cholesterol',
      render: (value) => value ? `${value.toFixed(0)} mg/dL` : 'N/A',
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <p style={{ marginTop: '16px' }}>Loading patient data...</p>
      </div>
    );
  }

  if (!patient) {
    return (
      <Alert
        message="Patient Not Found"
        description="The requested patient could not be found."
        type="error"
        showIcon
      />
    );
  }

  const latestPrediction = predictions.length > 0 ? predictions[0] : null;

  return (
    <div>
      {/* Patient Header */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <UserOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
          </Col>
          <Col flex="auto">
            <Title level={2} style={{ margin: 0 }}>
              {patient.name}
            </Title>
            <Space size="large">
              <Text>Age: {patient.age}</Text>
              <Text>Gender: {patient.gender === 'M' ? 'Male' : 'Female'}</Text>
              <Text>Patient ID: {patient.id}</Text>
            </Space>
          </Col>
          <Col>
            <Space direction="vertical" size="small">
              <Space>
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  size="large"
                  loading={predicting}
                  onClick={generatePrediction}
                  disabled={healthRecords.length < 1}
                >
                  Generate Prediction
                </Button>
                <Button
                  icon={<UploadOutlined />}
                  size="large"
                  onClick={() => setUploadModalVisible(true)}
                >
                  Upload Report
                </Button>
              </Space>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {healthRecords.length < 1 
                  ? `Need at least 1 record (current: ${healthRecords.length})` 
                  : `Ready with ${healthRecords.length} records`
                }
              </Text>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Latest Prediction */}
      {latestPrediction && (
        <Card 
          title={
            <Space>
              <HeartOutlined />
              Latest Risk Assessment
            </Space>
          }
          style={{ marginBottom: '24px' }}
        >
          <Row gutter={[16, 16]}>
            <Col xs={24} md={8}>
              <Card size="small">
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={Math.round(latestPrediction.diabetes_risk * 100)}
                    strokeColor={getRiskColor(latestPrediction.diabetes_risk)}
                    size={120}
                  />
                  <Title level={4} style={{ marginTop: '16px' }}>Diabetes Risk</Title>
                  <Tag color={getRiskColor(latestPrediction.diabetes_risk)}>
                    {getRiskLevel(latestPrediction.diabetes_risk)} Risk
                  </Tag>
                </div>
              </Card>
            </Col>
            
            <Col xs={24} md={8}>
              <Card size="small">
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={Math.round(latestPrediction.heart_disease_risk * 100)}
                    strokeColor={getRiskColor(latestPrediction.heart_disease_risk)}
                    size={120}
                  />
                  <Title level={4} style={{ marginTop: '16px' }}>Heart Disease Risk</Title>
                  <Tag color={getRiskColor(latestPrediction.heart_disease_risk)}>
                    {getRiskLevel(latestPrediction.heart_disease_risk)} Risk
                  </Tag>
                </div>
              </Card>
            </Col>
            
            <Col xs={24} md={8}>
              <Card size="small">
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={Math.round(latestPrediction.kidney_disease_risk * 100)}
                    strokeColor={getRiskColor(latestPrediction.kidney_disease_risk)}
                    size={120}
                  />
                  <Title level={4} style={{ marginTop: '16px' }}>Kidney Disease Risk</Title>
                  <Tag color={getRiskColor(latestPrediction.kidney_disease_risk)}>
                    {getRiskLevel(latestPrediction.kidney_disease_risk)} Risk
                  </Tag>
                </div>
              </Card>
            </Col>
          </Row>

          {/* High Risk Warning */}
          {(latestPrediction.diabetes_risk > 0.7 || 
            latestPrediction.heart_disease_risk > 0.7 || 
            latestPrediction.kidney_disease_risk > 0.7) && (
            <Alert
              message="High Risk Alert"
              description="This patient shows high risk for one or more conditions. Consider immediate clinical review."
              type="warning"
              showIcon
              icon={<WarningOutlined />}
              style={{ marginTop: '16px' }}
            />
          )}

          {/* Explanation */}
          <Divider />
          <Title level={4}>AI Analysis</Title>
          <div style={{ 
            background: '#f5f5f5', 
            padding: '20px', 
            borderRadius: '8px',
            lineHeight: '1.8'
          }}>
            {latestPrediction.explanation.split('\n').map((line, index) => {
              if (line.trim().startsWith('-')) {
                return (
                  <div key={index} style={{ marginLeft: '20px', marginBottom: '8px' }}>
                    <Text>• {line.trim().substring(1)}</Text>
                  </div>
                );
              } else if (line.trim()) {
                return (
                  <div key={index} style={{ marginBottom: '12px' }}>
                    <Text strong={line.includes(':') && !line.includes('Risk')}>
                      {line.trim()}
                    </Text>
                  </div>
                );
              }
              return null;
            })}
          </div>
        </Card>
      )}

      {/* Health Trends Chart */}
      {chartData.length > 0 && (
        <Card title="Health Trends" style={{ marginBottom: '24px' }}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="glucose" stroke="#8884d8" name="Glucose (mg/dL)" />
              <Line type="monotone" dataKey="systolic_bp" stroke="#82ca9d" name="Systolic BP (mmHg)" />
              <Line type="monotone" dataKey="creatinine" stroke="#ffc658" name="Creatinine (mg/dL)" />
              <Line type="monotone" dataKey="cholesterol" stroke="#ff7300" name="Cholesterol (mg/dL)" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Health Records Table */}
      <Card 
        title={
          <Space>
            <CalendarOutlined />
            Health Records ({healthRecords.length} visits)
          </Space>
        }
      >
        {healthRecords.length === 0 ? (
          <Alert
            message="No Health Records"
            description="No health records found for this patient. Add health records to generate predictions."
            type="info"
            showIcon
          />
        ) : (
          <Table
            columns={healthRecordColumns}
            dataSource={healthRecords}
            rowKey="id"
            pagination={{ pageSize: 10 }}
            scroll={{ x: 800 }}
          />
        )}
      </Card>

      {/* Upload Modal */}
      <Modal
        title={
          <Space>
            <FileTextOutlined />
            Upload Medical Report
          </Space>
        }
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={600}
      >
        <div style={{ padding: '20px 0' }}>
          <Alert
            message="Supported File Types"
            description="PDF files, JPG, PNG, TIFF, BMP images. The system will extract medical values and merge them with existing records without overwriting manual entries."
            type="info"
            showIcon
            style={{ marginBottom: '20px' }}
          />
          
          <Upload.Dragger
            name="file"
            multiple={false}
            beforeUpload={handleFileUpload}
            showUploadList={false}
            accept=".pdf,.jpg,.jpeg,.png,.tiff,.bmp"
            disabled={uploading}
          >
            <p className="ant-upload-drag-icon">
              <UploadOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
            </p>
            <p className="ant-upload-text">
              {uploading ? 'Processing report...' : 'Click or drag medical report to upload'}
            </p>
            <p className="ant-upload-hint">
              Supports PDF and image files (JPG, PNG, TIFF, BMP)
            </p>
          </Upload.Dragger>
          
          {uploading && (
            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <Spin size="large" />
              <p style={{ marginTop: '10px' }}>Extracting medical data from report...</p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
};

export default PatientDetail;