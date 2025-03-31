import { Row, Col, Typography } from 'antd';

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

const resourceData = [
  { label: 'GPUs', value: '0' },
  { label: 'vCPUs', value: '0' },
  { label: 'Storage', value: '0 GB' },
  { label: 'Endpoints', value: '0' },
];

const ResourceSection: React.FC = () => {
  return (
    <Col span={6}>
      <Row style={styles.sectionTitle}>
        <Col>
          <Title level={4}>Resources</Title>
          <Text>Monitor your GPU, vCPU, storage, and endpoint usage.</Text>
        </Col>
      </Row>
      <Row gutter={24}>
        {resourceData.map((item, index) => (
          <Col span={12} key={index} style={{ marginBottom: 30 }}>
            <div style={styles.label}>{item.label}</div>
            <div style={styles.value}>{item.value}</div>
          </Col>
        ))}
      </Row>
    </Col>
  );
};

export default ResourceSection; 