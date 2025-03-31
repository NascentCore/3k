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
    fontSize: '24px',
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
    <Row gutter={[24, 24]}>
      {featureCards.map((card, index) => (
        <Col span={6} key={index}>
          <Card hoverable style={styles.card}>
            <Card.Meta avatar={card.icon} title={card.title} description={card.description} />
          </Card>
        </Col>
      ))}
    </Row>
  );
};

export default FeatureCards; 