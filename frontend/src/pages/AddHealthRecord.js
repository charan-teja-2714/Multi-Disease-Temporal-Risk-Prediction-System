import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Card,
  Form,
  InputNumber,
  Select,
  DatePicker,
  Button,
  Row,
  Col,
  message,
  Typography,
  Divider,
  Space,
  Alert,
  Upload,
  Modal
} from 'antd';
import { SaveOutlined, UserOutlined, UploadOutlined } from '@ant-design/icons';
import { patientAPI } from '../services/api';
import moment from 'moment';
import axios from 'axios';

const { Title, Text } = Typography;
const { Option } = Select;

const AddHealthRecord = () => {
  const [form] = Form.useForm();
  const [searchParams] = useSearchParams();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [showExtractModal, setShowExtractModal] = useState(false);
  const [extractedData, setExtractedData] = useState({});

  useEffect(() => {
    loadPatients();
  }, []);

  // Pre-select patient when coming from /add-record?patient=<id>
  useEffect(() => {
    const patientId = searchParams.get('patient');
    if (patientId && patients.length > 0) {
      form.setFieldValue('patient_id', parseInt(patientId));
    }
  }, [searchParams, patients, form]);

  const loadPatients = async () => {
    try {
      setLoading(true);
      const data = await patientAPI.getPatients();
      setPatients(data);
    } catch (error) {
      message.error('Failed to load patients');
      console.error('Error loading patients:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (values) => {
    try {
      setSubmitting(true);
      
      const recordData = {
        ...values,
        visit_date: values.visit_date.format('YYYY-MM-DD'),
      };

      await patientAPI.addHealthRecord(recordData);
      message.success('Health record added successfully');
      form.resetFields();
      form.setFieldsValue({ visit_date: moment() });
      
    } catch (error) {
      message.error('Failed to add health record');
      console.error('Error adding health record:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleFileUpload = async (file, patientId) => {
    if (!patientId) {
      message.error('Please select a patient first');
      return false;
    }

    try {
      setExtracting(true);
      const formData = new FormData();
      formData.append('file', file);

      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await axios.post(
        `${API_BASE_URL}/extract-report/${patientId}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (response.data.success) {
        setExtractedData(response.data.extracted_values);
        setShowExtractModal(true);
        message.success(`Extracted ${response.data.fields_found} values from report`);
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to extract data from report');
    } finally {
      setExtracting(false);
    }
    return false;
  };

  const handleConfirmExtracted = async () => {
    try {
      setSubmitting(true);
      const patientId = form.getFieldValue('patient_id');
      const visitDate = form.getFieldValue('visit_date');

      const recordData = {
        patient_id: patientId,
        visit_date: visitDate.format('YYYY-MM-DD'),
        ...extractedData
      };

      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      await axios.post(`${API_BASE_URL}/save-extracted-record/${patientId}`, recordData);
      
      message.success('Health record saved successfully');
      setShowExtractModal(false);
      setExtractedData({});
      form.resetFields();
      form.setFieldsValue({ visit_date: moment() });
    } catch (error) {
      message.error('Failed to save health record');
    } finally {
      setSubmitting(false);
    }
  };

  const normalRanges = {
    glucose: { min: 70, max: 100, unit: 'mg/dL' },
    hba1c: { min: 4.0, max: 5.6, unit: '%' },
    creatinine: { min: 0.6, max: 1.2, unit: 'mg/dL' },
    bun: { min: 7, max: 20, unit: 'mg/dL' },
    systolic_bp: { min: 90, max: 120, unit: 'mmHg' },
    diastolic_bp: { min: 60, max: 80, unit: 'mmHg' },
    cholesterol: { min: 125, max: 200, unit: 'mg/dL' },
    hdl: { min: 40, max: 60, unit: 'mg/dL' },
    ldl: { min: 100, max: 130, unit: 'mg/dL' },
    triglycerides: { min: 50, max: 150, unit: 'mg/dL' },
    bmi: { min: 18.5, max: 24.9, unit: 'kg/m²' }
  };

  const renderInputWithRange = (name, label, range) => (
    <Form.Item
      name={name}
      label={
        <Space direction="vertical" size={0}>
          <Text strong>{label}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Normal: {range.min}-{range.max} {range.unit}
          </Text>
        </Space>
      }
    >
      <InputNumber
        placeholder={`Enter ${label.toLowerCase()}`}
        style={{ width: '100%' }}
        min={0}
        step={name === 'hba1c' || name === 'creatinine' || name === 'bmi' ? 0.1 : 1}
        precision={name === 'hba1c' || name === 'creatinine' || name === 'bmi' ? 1 : 0}
        addonAfter={range.unit}
      />
    </Form.Item>
  );

  return (
    <div>
      <Title level={2}>Add Health Record</Title>
      <Text type="secondary">
        Enter health measurements manually or upload a medical report (PDF/Image).
      </Text>

      <Card style={{ marginTop: '24px' }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            visit_date: moment()
          }}
        >
          {/* Patient Selection */}
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Form.Item
                name="patient_id"
                label="Select Patient"
                rules={[{ required: true, message: 'Please select a patient' }]}
              >
                <Select
                  placeholder="Choose a patient"
                  loading={loading}
                  showSearch
                  filterOption={(input, option) => {
                    const patient = patients.find(p => p.id === option.value);
                    if (!patient) return false;
                    return patient.name.toLowerCase().indexOf(input.toLowerCase()) >= 0;
                  }}
                >
                  {patients.map(patient => (
                    <Option key={patient.id} value={patient.id}>
                      <Space>
                        <UserOutlined />
                        {patient.name} (Age: {patient.age}, {patient.gender === 'M' ? 'Male' : 'Female'})
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} md={12}>
              <Form.Item
                name="visit_date"
                label="Visit Date"
                rules={[{ required: true, message: 'Please select visit date' }]}
              >
                <DatePicker
                  style={{ width: '100%' }}
                  format="YYYY-MM-DD"
                  disabledDate={(current) => current && current > moment().endOf('day')}
                />
              </Form.Item>
            </Col>
          </Row>

          <Alert
            message="Upload Medical Report"
            description={
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>Upload PDF or image report to auto-extract health values</Text>
                <Upload
                  beforeUpload={(file) => handleFileUpload(file, form.getFieldValue('patient_id'))}
                  accept=".pdf,.jpg,.jpeg,.png,.tiff,.bmp"
                  showUploadList={false}
                >
                  <Button icon={<UploadOutlined />} loading={extracting}>
                    Upload Report
                  </Button>
                </Upload>
              </Space>
            }
            type="info"
            showIcon
            style={{ marginBottom: '24px' }}
          />

          <Divider>Blood Sugar & Diabetes Markers</Divider>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('glucose', 'Glucose', normalRanges.glucose)}
            </Col>
            <Col xs={24} md={12}>
              {renderInputWithRange('hba1c', 'HbA1c', normalRanges.hba1c)}
            </Col>
          </Row>

          <Divider>Kidney Function Markers</Divider>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('creatinine', 'Creatinine', normalRanges.creatinine)}
            </Col>
            <Col xs={24} md={12}>
              {renderInputWithRange('bun', 'BUN (Blood Urea Nitrogen)', normalRanges.bun)}
            </Col>
          </Row>

          <Divider>Blood Pressure</Divider>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('systolic_bp', 'Systolic Blood Pressure', normalRanges.systolic_bp)}
            </Col>
            <Col xs={24} md={12}>
              {renderInputWithRange('diastolic_bp', 'Diastolic Blood Pressure', normalRanges.diastolic_bp)}
            </Col>
          </Row>

          <Divider>Cholesterol Panel</Divider>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('cholesterol', 'Total Cholesterol', normalRanges.cholesterol)}
            </Col>
            <Col xs={24} md={12}>
              {renderInputWithRange('hdl', 'HDL (Good Cholesterol)', normalRanges.hdl)}
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('ldl', 'LDL (Bad Cholesterol)', normalRanges.ldl)}
            </Col>
            <Col xs={24} md={12}>
              {renderInputWithRange('triglycerides', 'Triglycerides', normalRanges.triglycerides)}
            </Col>
          </Row>

          <Divider>Physical Measurements</Divider>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              {renderInputWithRange('bmi', 'BMI (Body Mass Index)', normalRanges.bmi)}
            </Col>
          </Row>

          <Alert
            message="Data Quality Tips"
            description="Complete health records improve prediction accuracy. Missing values are handled automatically, but try to include key markers like glucose, blood pressure, and creatinine when available."
            type="info"
            showIcon
            style={{ margin: '24px 0' }}
          />

          <Form.Item style={{ textAlign: 'right', marginBottom: 0 }}>
            <Space>
              <Button onClick={() => {
                form.resetFields();
                form.setFieldsValue({ visit_date: moment() });
              }}>
                Clear Form
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SaveOutlined />}
                loading={submitting}
                size="large"
              >
                Save Health Record
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* Extracted Data Confirmation Modal */}
      <Modal
        title="Confirm Extracted Values"
        open={showExtractModal}
        onOk={handleConfirmExtracted}
        onCancel={() => setShowExtractModal(false)}
        okText="Save Record"
        cancelText="Cancel"
        confirmLoading={submitting}
        width={600}
      >
        <Text type="secondary" style={{ display: 'block', marginBottom: '16px' }}>
          Review and edit the extracted values before saving:
        </Text>
        <Form layout="vertical">
          <Row gutter={[16, 16]}>
            {Object.entries(extractedData).map(([key, value]) => (
              <Col xs={24} md={12} key={key}>
                <Form.Item label={key.replace('_', ' ').toUpperCase()}>
                  <InputNumber
                    value={value}
                    onChange={(val) => setExtractedData({ ...extractedData, [key]: val })}
                    style={{ width: '100%' }}
                    step={key === 'hba1c' || key === 'creatinine' || key === 'bmi' ? 0.1 : 1}
                  />
                </Form.Item>
              </Col>
            ))}
          </Row>
        </Form>
      </Modal>
    </div>
  );
};

export default AddHealthRecord;