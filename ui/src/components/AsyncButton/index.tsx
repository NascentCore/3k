import React, { useState } from 'react';
import { Button, ButtonProps } from 'antd';

interface AsyncButtonProps extends ButtonProps {
  onClick: () => Promise<void>;
  children: React.ReactNode;
}

const AsyncButton: React.FC<AsyncButtonProps> = ({ onClick, children, ...rest }) => {
  const [loading, setLoading] = useState(false);

  const handleAsyncClick = async () => {
    setLoading(true);
    try {
      await onClick();
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button onClick={handleAsyncClick} loading={loading} {...rest}>
      {children}
    </Button>
  );
};

export default AsyncButton;
