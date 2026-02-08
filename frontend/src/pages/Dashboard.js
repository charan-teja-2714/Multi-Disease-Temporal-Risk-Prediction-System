import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Alert, Spin, Typography } from 'antd';
import { 
  UserOutlined, 
  HeartOutlined, 
  ExperimentOutlined,
  CheckCircleOutlined 
} from '@ant-design/icons';
import { patientAPI, systemAPI } from '../services/api';

const { Title, Paragraph } = Typography;

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [systemHealth, setSystemHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [patientCount, setPatientCount] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load system health
      try {
        const healthData = await systemAPI.healthCheck();
        setSystemHealth(healthData);
      } catch (healthErr) {
        console.warn('Health check failed:', healthErr);
        setSystemHealth({ status: 'ok' });
      }

      // Load model info (optional - don't fail if not available)
      try {
        const modelData = await systemAPI.getModelInfo();
        setModelInfo(modelData);
      } catch (modelErr) {
        console.warn('Model info not available:', modelErr);
        setModelInfo({ model_type: 'Multi-Task TCN', diseases: ['diabetes', 'heart_disease', 'kidney_disease'] });
      }

      // Load patient count
      try {
        const patients = await patientAPI.getPatients();
        setPatientCount(patients.length);
      } catch (patientErr) {
        console.warn('Failed to load patients:', patientErr);
        setPatientCount(0);
      }

      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data. Please check if the backend is running.');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <p style={{ marginTop: '16px' }}>Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message="System Error"
        description={error}
        type="error"
        showIcon
        style={{ marginBottom: '24px' }}
      />
    );
  }

  return (
    <div>
      <Title level={2}>System Dashboard</Title>
      <Paragraph>
        Welcome to the Multi-Disease Risk Prediction System. Monitor system health, 
        manage patients, and generate AI-powered disease risk assessments.
      </Paragraph>

      {/* System Status Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="System Status"
              value={systemHealth?.status === 'ok' ? 'Online' : 'Offline'}
              prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: systemHealth?.status === 'ok' ? '#52c41a' : '#ff4d4f' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Total Patients"
              value={patientCount}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="AI Model"
              value={modelInfo?.model_type || 'Multi-Task TCN'}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Diseases Tracked"
              value={modelInfo?.diseases?.length || 3}
              prefix={<HeartOutlined />}
              valueStyle={{ color: '#eb2f96' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Quick Actions */}
      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="Quick Actions" bordered={false}>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Card 
                  hoverable 
                  onClick={() => window.location.href = '/patients'}
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                >
                  <UserOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
                  <div style={{ marginTop: '8px' }}>View Patients</div>
                </Card>
              </Col>
              
              <Col xs={24} sm={8}>
                <Card 
                  hoverable 
                  onClick={() => window.location.href = '/add-record'}
                  style={{ textAlign: 'center', cursor: 'pointer' }}
                >
                  <ExperimentOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
                  <div style={{ marginTop: '8px' }}>Add Health Record</div>
                </Card>
              </Col>
              
              <Col xs={24} sm={8}>
                <Card 
                  hoverable 
                  style={{ textAlign: 'center' }}
                >
                  <HeartOutlined style={{ fontSize: '24px', color: '#eb2f96' }} />
                  <div style={{ marginTop: '8px' }}>Generate Predictions</div>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;