import { Row, Col, Typography } from 'antd';
import UsageChart from './UsageChart';

const { Title, Text } = Typography;

const styles = {
  label: {
    fontSize: '14px',
    fontWeight: '400',
    lineHeight: 1.75,
    color: 'rgba(0,0,0,0.75)',
  },
  value: {
    fontSize: '18px',
    fontWeight: '700',
    lineHeight: 1.5,
  },
  sectionTitle: {
    marginBottom: '30px',
  },
};

const usageData = [
  { label: 'Rolling Average', value: '$0.00 / day' },
  { label: 'Current Spend Rate', value: '$0.00 / hr' },
];

const UsageSection: React.FC = () => {
  return (
    <Col span={18}>
      <Row style={styles.sectionTitle}>
        <Col>
          <Title level={4}>Usage</Title>
          <Text>Keep an eye on your daily spend with real-time insights.</Text>
        </Col>
      </Row>
      <Row gutter={24}>
        {usageData.map((item, index) => (
          <Col span={12} key={index}>
            <div style={styles.label}>{item.label}</div>
            <div style={styles.value}>{item.value}</div>
          </Col>
        ))}
      </Row>
      <Row>
        <UsageChart />
      </Row>
    </Col>
  );
};

export default UsageSection; 