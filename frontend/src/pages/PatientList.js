import React, { useState, useEffect } from 'react';
import { 
  Table, 
  Button, 
  Modal, 
  Form, 
  Input, 
  Select, 
  InputNumber, 
  message,
  Space,
  Tag,
  Typography
} from 'antd';
import { PlusOutlined, EyeOutlined, UserOutlined } from '@ant-design/icons';
import { patientAPI } from '../services/api';
import moment from 'moment';

const { Title } = Typography;
const { Option } = Select;

const PatientList = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

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

  const handleAddPatient = async (values) => {
    try {
      await patientAPI.createPatient(values);
      message.success('Patient added successfully');
      setModalVisible(false);
      form.resetFields();
      loadPatients();
    } catch (error) {
      message.error('Failed to add patient');
      console.error('Error adding patient:', error);
    }
  };

  const viewPatientDetails = (patientId) => {
    window.location.href = `/patient/${patientId}`;
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text) => (
        <Space>
          <UserOutlined />
          {text}
        </Space>
      ),
    },
    {
      title: 'Age',
      dataIndex: 'age',
      key: 'age',
      width: 80,
    },
    {
      title: 'Gender',
      dataIndex: 'gender',
      key: 'gender',
      width: 100,
      render: (gender) => (
        <Tag color={gender === 'M' ? 'blue' : 'pink'}>
          {gender === 'M' ? 'Male' : 'Female'}
        </Tag>
      ),
    },
    {
      title: 'Created Date',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => moment(date).format('YYYY-MM-DD'),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            icon={<EyeOutlined />}
            size="small"
            onClick={() => viewPatientDetails(record.id)}
          >
            View
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '24px'
      }}>
        <Title level={2}>Patient Management</Title>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setModalVisible(true)}
        >
          Add New Patient
        </Button>
      </div>

      <Table
        columns={columns}
        dataSource={patients}
        loading={loading}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
          showTotal: (total, range) =>
            `${range[0]}-${range[1]} of ${total} patients`,
        }}
      />

      <Modal
        title="Add New Patient"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={500}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleAddPatient}
        >
          <Form.Item
            name="name"
            label="Patient Name"
            rules={[
              { required: true, message: 'Please enter patient name' },
              { min: 2, message: 'Name must be at least 2 characters' }
            ]}
          >
            <Input placeholder="Enter patient full name" />
          </Form.Item>

          <Form.Item
            name="age"
            label="Age"
            rules={[
              { required: true, message: 'Please enter patient age' },
              { type: 'number', min: 1, max: 120, message: 'Age must be between 1 and 120' }
            ]}
          >
            <InputNumber
              placeholder="Enter age"
              style={{ width: '100%' }}
              min={1}
              max={120}
            />
          </Form.Item>

          <Form.Item
            name="gender"
            label="Gender"
            rules={[{ required: true, message: 'Please select gender' }]}
          >
            <Select placeholder="Select gender">
              <Option value="M">Male</Option>
              <Option value="F">Female</Option>
            </Select>
          </Form.Item>

          <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
            <Space>
              <Button onClick={() => {
                setModalVisible(false);
                form.resetFields();
              }}>
                Cancel
              </Button>
              <Button type="primary" htmlType="submit">
                Add Patient
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default PatientList;