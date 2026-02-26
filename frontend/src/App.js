import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography, Button, Avatar, Space } from 'antd';
import {
  UserOutlined,
  HeartOutlined,
  DashboardOutlined,
  PlusOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import Dashboard from './pages/Dashboard';
import PatientList from './pages/PatientList';
import PatientDetail from './pages/PatientDetail';
import AddHealthRecord from './pages/AddHealthRecord';
import Login from './pages/Login';
import Signup from './pages/Signup';
import './App.css';

const { Header, Sider, Content } = Layout;
const { Title, Text } = Typography;

function AppContent({ user, onLogout }) {
  const [collapsed, setCollapsed] = React.useState(false);
  const location = useLocation();

  const menuItems = [
    { key: '1', icon: <DashboardOutlined />, label: 'Dashboard', path: '/' },
    { key: '2', icon: <UserOutlined />, label: 'Patients', path: '/patients' },
    { key: '3', icon: <PlusOutlined />, label: 'Add Health Record', path: '/add-record' }
  ];

  const getSelectedKey = () => {
    const p = location.pathname;
    if (p === '/') return ['1'];
    if (p === '/patients') return ['2'];
    if (p === '/add-record') return ['3'];
    if (p.startsWith('/patient/')) return ['2'];
    return ['1'];
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed} theme="light">
        <div style={{ padding: '16px', textAlign: 'center', borderBottom: '1px solid #f0f0f0' }}>
          <HeartOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
          {!collapsed && (
            <Title level={4} style={{ margin: '8px 0 0 0', color: '#1890ff' }}>
              MedPredict
            </Title>
          )}
        </div>

        <Menu
          theme="light"
          selectedKeys={getSelectedKey()}
          mode="inline"
          items={menuItems.map(item => ({
            key: item.key,
            icon: item.icon,
            label: <Link to={item.path}>{item.label}</Link>
          }))}
        />
      </Sider>

      <Layout className="site-layout">
        <Header
          className="site-layout-background"
          style={{
            padding: '0 24px',
            background: '#fff',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            Multi-Disease Risk Prediction System
          </Title>

          <Space>
            <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#1890ff' }} />
            <Text strong>{user.username}</Text>
            <Button
              icon={<LogoutOutlined />}
              onClick={onLogout}
              type="text"
            >
              Logout
            </Button>
          </Space>
        </Header>

        <Content
          className="site-layout-background"
          style={{ margin: '24px 16px', padding: 24, minHeight: 280, background: '#fff' }}
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/patients" element={<PatientList />} />
            <Route path="/patient/:id" element={<PatientDetail />} />
            <Route path="/add-record" element={<AddHealthRecord />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}

function App() {
  // sessionStorage: survives page refresh but clears when browser/tab is closed
  const [user, setUser] = useState(() => {
    try {
      const stored = sessionStorage.getItem('user');
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [showSignup, setShowSignup] = useState(false);

  const handleLogin = (userData) => {
    sessionStorage.setItem('user', JSON.stringify(userData));
    setUser(userData);
  };

  // After signup → go to login page instead of auto-logging in
  const handleSignup = () => {
    setShowSignup(false);
  };

  const handleLogout = () => {
    sessionStorage.removeItem('user');
    setUser(null);
    setShowSignup(false);
  };

  if (!user) {
    if (showSignup) {
      return (
        <Signup
          onSignup={handleSignup}
          onGoToLogin={() => setShowSignup(false)}
        />
      );
    }
    return (
      <Login
        onLogin={handleLogin}
        onGoToSignup={() => setShowSignup(true)}
      />
    );
  }

  return (
    <Router>
      <AppContent user={user} onLogout={handleLogout} />
    </Router>
  );
}

export default App;
