import React, { useState, useEffect } from 'react';
import { Card, Form, Select, DatePicker, Button, Upload, message, Typography, Space, Alert } from 'antd';
import { UploadOutlined, UserOutlined, FileTextOutlined } from '@ant-design/icons';
import { patientAPI } from '../services/api';
import moment from 'moment';
import axios from 'axios';

const { Title, Text } = Typography;
const { Option } = Select;

const UploadReport = () => {
  const [form] = Form.useForm();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileList, setFileList] = useState([]);

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
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (values) => {
    if (fileList.length === 0) {
      message.error('Please select a PDF file');
      return;
    }

    try {
      setUploading(true);
      
      const formData = new FormData();
      formData.append('file', fileList[0].originFileObj);
      formData.append('visit_date', values.visit_date.format('YYYY-MM-DD'));

      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await axios.post(
        `${API_BASE_URL}/upload-report/${values.patient_id}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (response.data.success) {
        message.success(`Report uploaded! Extracted ${response.data.fields_found} health values`);
        form.resetFields();
        setFileList([]);
        form.setFieldsValue({ visit_date: moment() });
      } else {
        message.warning(response.data.message || 'No data extracted from report');
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to upload report');
    } finally {
      setUploading(false);
    }
  };

  const uploadProps = {
    beforeUpload: (file) => {
      if (!file.name.endsWith('.pdf')) {
        message.error('Only PDF files are supported');
        return false;
      }
      setFileList([file]);
      return false;
    },
    fileList,
    onRemove: () => setFileList([]),
    maxCount: 1,
  };

  return (
    <div>
      <Title level={2}>Upload Medical Report</Title>
      <Text type="secondary">
        Upload PDF medical reports to automatically extract health data
      </Text>

      <Card style={{ marginTop: '24px', maxWidth: '800px' }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{ visit_date: moment() }}
        >
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
                return patient?.name.toLowerCase().includes(input.toLowerCase());
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

          <Form.Item
            name="visit_date"
            label="Report Date"
            rules={[{ required: true, message: 'Please select report date' }]}
          >
            <DatePicker
              style={{ width: '100%' }}
              format="YYYY-MM-DD"
              disabledDate={(current) => current && current > moment().endOf('day')}
            />
          </Form.Item>

          <Form.Item
            label="Upload PDF Report"
            required
          >
            <Upload {...uploadProps}>
              <Button icon={<UploadOutlined />}>Select PDF File</Button>
            </Upload>
          </Form.Item>

          <Alert
            message="Supported Report Formats"
            description={
              <div>
                <Text>The system can extract values with different naming conventions:</Text>
                <ul style={{ marginTop: '8px', marginBottom: 0 }}>
                  <li>Glucose: glucose, blood glucose, FBS, blood sugar</li>
                  <li>HbA1c: HbA1c, A1C, glycated hemoglobin</li>
                  <li>Creatinine: creatinine, serum creatinine, Cr</li>
                  <li>BUN: BUN, blood urea nitrogen, urea</li>
                  <li>Blood Pressure: BP 120/80, systolic/diastolic</li>
                  <li>Cholesterol: total cholesterol, HDL, LDL, triglycerides</li>
                </ul>
              </div>
            }
            type="info"
            showIcon
            icon={<FileTextOutlined />}
            style={{ marginBottom: '24px' }}
          />

          <Form.Item style={{ textAlign: 'right', marginBottom: 0 }}>
            <Space>
              <Button onClick={() => {
                form.resetFields();
                setFileList([]);
                form.setFieldsValue({ visit_date: moment() });
              }}>
                Clear
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                icon={<UploadOutlined />}
                loading={uploading}
                size="large"
              >
                Upload Report
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default UploadReport;
