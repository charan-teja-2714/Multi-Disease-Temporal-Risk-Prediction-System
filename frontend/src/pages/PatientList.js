import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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
  Typography,
  Popconfirm,
  Tooltip,
} from 'antd';
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  UserOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { patientAPI } from '../services/api';
import moment from 'moment';

const { Title } = Typography;
const { Option } = Select;

const PatientList = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [modalVisible, setModalVisible] = useState(false);
  const [editingPatient, setEditingPatient] = useState(null); // null = add mode
  const [form] = Form.useForm();

  useEffect(() => {
    loadPatients();
  }, []);

  // Filter by search whenever list or search term changes
  useEffect(() => {
    const q = search.trim().toLowerCase();
    setFiltered(q ? patients.filter(p => p.name.toLowerCase().includes(q)) : patients);
  }, [patients, search]);

  const loadPatients = async () => {
    try {
      setLoading(true);
      const data = await patientAPI.getPatients();
      setPatients(data);
    } catch {
      message.error('Failed to load patients');
    } finally {
      setLoading(false);
    }
  };

  const openAddModal = () => {
    setEditingPatient(null);
    form.resetFields();
    setModalVisible(true);
  };

  const openEditModal = (patient) => {
    setEditingPatient(patient);
    form.setFieldsValue({ name: patient.name, age: patient.age, gender: patient.gender });
    setModalVisible(true);
  };

  const closeModal = () => {
    setModalVisible(false);
    setEditingPatient(null);
    form.resetFields();
  };

  const handleSubmit = async (values) => {
    try {
      if (editingPatient) {
        await patientAPI.updatePatient(editingPatient.id, values);
        message.success('Patient updated successfully');
      } else {
        await patientAPI.createPatient(values);
        message.success('Patient added successfully');
      }
      closeModal();
      loadPatients();
    } catch (err) {
      message.error(err.response?.data?.detail || 'Operation failed');
    }
  };

  const handleDelete = async (patientId) => {
    try {
      await patientAPI.deletePatient(patientId);
      message.success('Patient deleted');
      loadPatients();
    } catch {
      message.error('Failed to delete patient');
    }
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 70,
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space
          style={{ cursor: 'pointer', color: '#1890ff' }}
          onClick={() => navigate(`/patient/${record.id}`)}
        >
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
      title: 'Added',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => moment(date).format('YYYY-MM-DD'),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      render: (_, record) => (
        <Space>
          <Tooltip title="View details">
            <Button
              type="primary"
              icon={<EyeOutlined />}
              size="small"
              onClick={() => navigate(`/patient/${record.id}`)}
            >
              View
            </Button>
          </Tooltip>
          <Tooltip title="Edit patient">
            <Button
              icon={<EditOutlined />}
              size="small"
              onClick={() => openEditModal(record)}
            >
              Edit
            </Button>
          </Tooltip>
          <Popconfirm
            title="Delete patient?"
            description="This will also delete all their health records and predictions."
            onConfirm={() => handleDelete(record.id)}
            okText="Delete"
            cancelText="Cancel"
            okButtonProps={{ danger: true }}
          >
            <Tooltip title="Delete patient">
              <Button danger icon={<DeleteOutlined />} size="small" />
            </Tooltip>
          </Popconfirm>
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
        marginBottom: '16px',
      }}>
        <Title level={2} style={{ margin: 0 }}>Patient Management</Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={openAddModal}>
          Add New Patient
        </Button>
      </div>

      {/* Search bar */}
      <Input
        prefix={<SearchOutlined />}
        placeholder="Search patients by name..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        allowClear
        style={{ marginBottom: '16px', maxWidth: 320 }}
      />

      <Table
        columns={columns}
        dataSource={filtered}
        loading={loading}
        rowKey="id"
        onRow={(record) => ({
          style: { cursor: 'pointer' },
        })}
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total, range) =>
            `${range[0]}-${range[1]} of ${total} patients`,
        }}
      />

      {/* Add / Edit Modal */}
      <Modal
        title={editingPatient ? 'Edit Patient' : 'Add New Patient'}
        open={modalVisible}
        onCancel={closeModal}
        footer={null}
        width={500}
        destroyOnClose
      >
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            name="name"
            label="Patient Name"
            rules={[
              { required: true, message: 'Please enter patient name' },
              { min: 2, message: 'Name must be at least 2 characters' },
            ]}
          >
            <Input placeholder="Enter full name" />
          </Form.Item>

          <Form.Item
            name="age"
            label="Age"
            rules={[
              { required: true, message: 'Please enter age' },
              { type: 'number', min: 1, max: 120, message: 'Age must be 1–120' },
            ]}
          >
            <InputNumber placeholder="Age" style={{ width: '100%' }} min={1} max={120} />
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
              <Button onClick={closeModal}>Cancel</Button>
              <Button type="primary" htmlType="submit">
                {editingPatient ? 'Save Changes' : 'Add Patient'}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default PatientList;
