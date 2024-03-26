import React, { useState } from 'react';
import { Typography, Segmented } from 'antd';
import { useIntl } from '@umijs/max';
import UserRegist from './UserRegist';
import SourceRegist from './SourceRegist';
const { Title } = Typography;

const RegistrationForm: React.FC = ({ setType }) => {
  const intl = useIntl();
  const type1 = intl.formatMessage({
    id: 'pages.regist.type.user',
    // defaultMessage: '算力用户注册',
  });
  const type2 = intl.formatMessage({
    id: 'pages.regist.type.source',
    // defaultMessage: '算力源注册',
  });
  const [rType, setRType] = useState(type1);
  return (
    <div
      style={{
        width: 330,
        margin: 'auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <Segmented
        size="large"
        options={[type1, type2]}
        style={{ marginBottom: 30 }}
        onChange={(value) => {
          setRType(value);
          console.log(value); // string
        }}
      />
      {rType === type1 && <UserRegist setType={setType} />}
      {rType === type2 && <SourceRegist setType={setType} />}
    </div>
  );
};

export default RegistrationForm;
