import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
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
  Modal,
  InputNumber,
  Popconfirm,
  Tooltip,
} from 'antd';
import {
  UserOutlined,
  HeartOutlined,
  ExperimentOutlined,
  CalendarOutlined,
  WarningOutlined,
  UploadOutlined,
  FileTextOutlined,
  ArrowLeftOutlined,
  DeleteOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { patientAPI } from '../services/api';
import moment from 'moment';

const { Title, Text } = Typography;

const PatientDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [healthRecords, setHealthRecords] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [extractedData, setExtractedData] = useState({});
  const [showConfirmModal, setShowConfirmModal] = useState(false);

  const loadPatientData = useCallback(async () => {
    try {
      setLoading(true);
      const patientData = await patientAPI.getPatient(id);
      setPatient(patientData);
      const recordsData = await patientAPI.getHealthRecords(id);
      setHealthRecords(recordsData);
      const predictionsData = await patientAPI.getPredictions(id);
      setPredictions(predictionsData);
    } catch (error) {
      message.error('Failed to load patient data');
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
      loadPatientData();
    } catch (error) {
      message.error('Failed to generate prediction. Ensure patient has at least 1 health record.');
    } finally {
      setPredicting(false);
    }
  };

  const handleDeleteRecord = async (recordId) => {
    try {
      await patientAPI.deleteHealthRecord(recordId);
      message.success('Health record deleted');
      loadPatientData();
    } catch {
      message.error('Failed to delete health record');
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setUploading(true);
      const result = await patientAPI.uploadMedicalReport(id, file);
      if (result.success) {
        setExtractedData(result.extracted_values || {});
        setShowConfirmModal(true);
        setUploadModalVisible(false);
        message.success(`Extracted ${result.fields_found || 0} values from report`);
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to extract data from report');
    } finally {
      setUploading(false);
    }
    return false;
  };

  const handleSaveExtracted = async () => {
    try {
      setUploading(true);
      const recordData = {
        patient_id: parseInt(id),
        visit_date: moment().format('YYYY-MM-DD'),
        source: 'report',
        ...extractedData,
      };
      await patientAPI.addHealthRecord(recordData);
      message.success('Health record saved successfully');
      setShowConfirmModal(false);
      setExtractedData({});
      loadPatientData();
    } catch {
      message.error('Failed to save health record');
    } finally {
      setUploading(false);
    }
  };

  const getRiskColor = (risk) => {
    if (risk < 0.3) return '#52c41a';
    if (risk < 0.7) return '#faad14';
    return '#ff4d4f';
  };

  const getRiskLevel = (risk) => {
    if (risk < 0.3) return 'Low';
    if (risk < 0.7) return 'Moderate';
    return 'High';
  };

  const chartData = healthRecords.map((record, index) => ({
    visit: `Visit ${index + 1}`,
    date: moment(record.visit_date).format('MM/DD'),
    glucose: record.glucose,
    systolic_bp: record.systolic_bp,
    creatinine: record.creatinine,
    cholesterol: record.cholesterol,
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
        <Tag color={source === 'report' ? 'blue' : 'green'}>
          {source === 'report' ? 'Report' : 'Manual'}
        </Tag>
      ),
    },
    {
      title: 'Glucose',
      dataIndex: 'glucose',
      key: 'glucose',
      render: (v) => v ? `${v.toFixed(1)} mg/dL` : 'N/A',
    },
    {
      title: 'HbA1c',
      dataIndex: 'hba1c',
      key: 'hba1c',
      render: (v) => v ? `${v.toFixed(1)}%` : 'N/A',
    },
    {
      title: 'BP (Sys/Dia)',
      key: 'bp',
      render: (_, r) =>
        r.systolic_bp && r.diastolic_bp
          ? `${r.systolic_bp.toFixed(0)}/${r.diastolic_bp.toFixed(0)} mmHg`
          : 'N/A',
    },
    {
      title: 'Creatinine',
      dataIndex: 'creatinine',
      key: 'creatinine',
      render: (v) => v ? `${v.toFixed(2)} mg/dL` : 'N/A',
    },
    {
      title: 'Cholesterol',
      dataIndex: 'cholesterol',
      key: 'cholesterol',
      render: (v) => v ? `${v.toFixed(0)} mg/dL` : 'N/A',
    },
    {
      title: 'Action',
      key: 'action',
      width: 80,
      render: (_, record) => (
        <Popconfirm
          title="Delete this record?"
          description="This health record will be permanently removed."
          onConfirm={() => handleDeleteRecord(record.id)}
          okText="Delete"
          cancelText="Cancel"
          okButtonProps={{ danger: true }}
        >
          <Tooltip title="Delete record">
            <Button danger icon={<DeleteOutlined />} size="small" />
          </Tooltip>
        </Popconfirm>
      ),
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
      {/* Back button */}
      <Button
        icon={<ArrowLeftOutlined />}
        onClick={() => navigate('/patients')}
        style={{ marginBottom: 16 }}
      >
        Back to Patients
      </Button>

      {/* Patient Header */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]} align="middle">
          <Col>
            <UserOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
          </Col>
          <Col flex="auto">
            <Title level={2} style={{ margin: 0 }}>{patient.name}</Title>
            <Space size="large">
              <Text>Age: {patient.age}</Text>
              <Text>Gender: {patient.gender === 'M' ? 'Male' : 'Female'}</Text>
              <Text type="secondary">Patient ID: #{patient.id}</Text>
            </Space>
          </Col>
          <Col>
            <Space direction="vertical" size="small">
              <Space wrap>
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
                  ? 'Need at least 1 health record'
                  : `Ready — ${healthRecords.length} record${healthRecords.length > 1 ? 's' : ''}`}
              </Text>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* Latest Prediction */}
      {latestPrediction && (
        <Card
          title={<Space><HeartOutlined />Latest Risk Assessment</Space>}
          style={{ marginBottom: '24px' }}
          extra={
            <Text type="secondary">
              {moment(latestPrediction.prediction_date).format('YYYY-MM-DD HH:mm')}
            </Text>
          }
        >
          <Row gutter={[16, 16]}>
            {[
              { label: 'Diabetes Risk', key: 'diabetes_risk' },
              { label: 'Heart Disease Risk', key: 'heart_disease_risk' },
              { label: 'Kidney Disease Risk', key: 'kidney_disease_risk' },
            ].map(({ label, key }) => (
              <Col xs={24} md={8} key={key}>
                <Card size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={Math.round(latestPrediction[key] * 100)}
                      strokeColor={getRiskColor(latestPrediction[key])}
                      size={120}
                    />
                    <Title level={4} style={{ marginTop: '16px' }}>{label}</Title>
                    <Tag color={getRiskColor(latestPrediction[key])}>
                      {getRiskLevel(latestPrediction[key])} Risk
                    </Tag>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>

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

          <Divider />
          <Title level={4}>AI Analysis</Title>
          <div style={{ background: '#f5f5f5', padding: '20px', borderRadius: '8px', lineHeight: '1.8' }}>
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
      {chartData.length > 1 && (
        <Card title="Health Trends Over Time" style={{ marginBottom: '24px' }}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <ChartTooltip />
              <Legend />
              <Line type="monotone" dataKey="glucose" stroke="#8884d8" name="Glucose (mg/dL)" dot />
              <Line type="monotone" dataKey="systolic_bp" stroke="#82ca9d" name="Systolic BP (mmHg)" dot />
              <Line type="monotone" dataKey="creatinine" stroke="#ffc658" name="Creatinine (mg/dL)" dot />
              <Line type="monotone" dataKey="cholesterol" stroke="#ff7300" name="Cholesterol (mg/dL)" dot />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Health Records Table */}
      <Card
        title={
          <Space>
            <CalendarOutlined />
            Health Records ({healthRecords.length} visit{healthRecords.length !== 1 ? 's' : ''})
          </Space>
        }
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate(`/add-record?patient=${id}`)}
          >
            Add Record
          </Button>
        }
      >
        {healthRecords.length === 0 ? (
          <Alert
            message="No Health Records"
            description="No health records found. Add a health record to generate predictions."
            type="info"
            showIcon
          />
        ) : (
          <Table
            columns={healthRecordColumns}
            dataSource={[...healthRecords].reverse()}
            rowKey="id"
            pagination={{ pageSize: 10 }}
            scroll={{ x: 900 }}
          />
        )}
      </Card>

      {/* Upload Modal */}
      <Modal
        title={<Space><FileTextOutlined />Upload Medical Report</Space>}
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={600}
      >
        <div style={{ padding: '20px 0' }}>
          <Alert
            message="Supported File Types"
            description="PDF, JPG, PNG, TIFF, BMP. The system extracts medical values and merges them without overwriting manual entries."
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
              {uploading ? 'Processing report...' : 'Click or drag medical report here'}
            </p>
            <p className="ant-upload-hint">Supports PDF and image files</p>
          </Upload.Dragger>
          {uploading && (
            <div style={{ textAlign: 'center', marginTop: '20px' }}>
              <Spin size="large" />
              <p style={{ marginTop: '10px' }}>Extracting medical data...</p>
            </div>
          )}
        </div>
      </Modal>

      {/* Confirm Extracted Data Modal */}
      <Modal
        title="Confirm Extracted Values"
        open={showConfirmModal}
        onOk={handleSaveExtracted}
        onCancel={() => setShowConfirmModal(false)}
        okText="Save Record"
        cancelText="Cancel"
        confirmLoading={uploading}
        width={700}
      >
        <Text type="secondary" style={{ display: 'block', marginBottom: '16px' }}>
          Review and edit the extracted values before saving:
        </Text>
        <Row gutter={[16, 16]}>
          {Object.entries(extractedData).map(([key, value]) => (
            <Col xs={24} md={12} key={key}>
              <div style={{ marginBottom: '12px' }}>
                <Text strong style={{ display: 'block', marginBottom: '4px' }}>
                  {key.replace(/_/g, ' ').toUpperCase()}
                </Text>
                <InputNumber
                  value={value}
                  onChange={(val) => setExtractedData({ ...extractedData, [key]: val })}
                  style={{ width: '100%' }}
                  step={['hba1c', 'creatinine', 'bmi'].includes(key) ? 0.1 : 1}
                />
              </div>
            </Col>
          ))}
        </Row>
      </Modal>
    </div>
  );
};

export default PatientDetail;
