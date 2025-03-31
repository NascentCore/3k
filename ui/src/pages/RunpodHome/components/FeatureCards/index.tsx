import { Row, Col, Card } from 'antd';
import {
  CloudServerOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  GiftOutlined,
} from '@ant-design/icons';

const styles = {
  card: {
    height: '100%',
  },
  icon: {
    fontSize: '20px',
  },
  title: {
    fontSize: '14px',
    marginBottom: '4px',
  },
  description: {
    fontSize: '12px',
    color: 'rgba(0,0,0,0.45)',
  },
};

const featureCards = [
  {
    icon: <CloudServerOutlined style={{ ...styles.icon, color: '#1890ff' }} />,
    title: 'GPU Cloud',
    description: 'Deploy a GPU pod.',
  },
  {
    icon: <ThunderboltOutlined style={{ ...styles.icon, color: '#52c41a' }} />,
    title: 'Serverless',
    description: 'Autoscale your workload with traffic with < 250ms cold-start.',
  },
  {
    icon: <DatabaseOutlined style={{ ...styles.icon, color: '#722ed1' }} />,
    title: 'Storage',
    description: 'Share network storage among all your pods.',
  },
  {
    icon: <GiftOutlined style={{ ...styles.icon, color: '#faad14' }} />,
    title: 'Earn Credits',
    description: 'Refer your friends & earn up to 6% for every penny they spend.',
  },
];

const FeatureCards: React.FC = () => {
  return (
    <Row gutter={[16, 16]}>
      {featureCards.map((card, index) => (
        <Col span={6} key={index}>
          <Card hoverable style={styles.card}>
            <Card.Meta 
              avatar={card.icon} 
              title={<div style={styles.title}>{card.title}</div>}
              description={<div style={styles.description}>{card.description}</div>}
            />
          </Card>
        </Col>
      ))}
    </Row>
  );
};

export default FeatureCards; 