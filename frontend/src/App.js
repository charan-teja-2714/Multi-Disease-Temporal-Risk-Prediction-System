import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import { 
  UserOutlined, 
  HeartOutlined, 
  DashboardOutlined,
  PlusOutlined 
} from '@ant-design/icons';
import Dashboard from './pages/Dashboard';
import PatientList from './pages/PatientList';
import PatientDetail from './pages/PatientDetail';
import AddHealthRecord from './pages/AddHealthRecord';
import './App.css';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

function AppContent() {
  const [collapsed, setCollapsed] = React.useState(false);
  const location = useLocation();

  const menuItems = [
    {
      key: '1',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
      path: '/'
    },
    {
      key: '2',
      icon: <UserOutlined />,
      label: 'Patients',
      path: '/patients'
    },
    {
      key: '3',
      icon: <PlusOutlined />,
      label: 'Add Health Record',
      path: '/add-record'
    }
  ];

  // Get current selected key based on location
  const getSelectedKey = () => {
    const currentPath = location.pathname;
    if (currentPath === '/') return ['1'];
    if (currentPath === '/patients') return ['2'];
    if (currentPath === '/add-record') return ['3'];
    if (currentPath.startsWith('/patient/')) return ['2']; // Patient detail belongs to patients section
    return ['1'];
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        theme="light"
      >
        <div className="logo" style={{ 
          padding: '16px', 
          textAlign: 'center',
          borderBottom: '1px solid #f0f0f0'
        }}>
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
            label: (
              <Link to={item.path} style={{ textDecoration: 'none' }}>
                {item.label}
              </Link>
            )
          }))}
        />
      </Sider>
      
      <Layout className="site-layout">
        <Header 
          className="site-layout-background" 
          style={{ 
            padding: '0 24px',
            background: '#fff',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}
        >
          <Title level={3} style={{ margin: '16px 0', color: '#1890ff' }}>
            Multi-Disease Risk Prediction System
          </Title>
        </Header>
        
        <Content
          className="site-layout-background"
          style={{
            margin: '24px 16px',
            padding: 24,
            minHeight: 280,
            background: '#fff'
          }}
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
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;