import React, { useState, useEffect } from 'react';
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
  Alert
} from 'antd';
import { SaveOutlined, UserOutlined } from '@ant-design/icons';
import { patientAPI } from '../services/api';
import moment from 'moment';

const { Title, Text } = Typography;
const { Option } = Select;

const AddHealthRecord = () => {
  const [form] = Form.useForm();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadPatients();
  }, []);

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
      
      // Format the data for API
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
        Enter health measurements for a patient visit. All fields are optional, 
        but more complete data improves prediction accuracy.
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
    </div>
  );
};

export default AddHealthRecord;