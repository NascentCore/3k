import { QuestionCircleOutlined } from '@ant-design/icons';
import { SelectLang as UmiSelectLang } from '@umijs/max';
import React, { useEffect } from 'react';
import { useIntl } from '@umijs/max';

export type SiderTheme = 'light' | 'dark';

export const SelectLang = () => {
  const intl = useIntl();

  const logo = intl.formatMessage({
    id: 'app.logo',
  });

  const title = intl.formatMessage({
    id: 'app.title',
  });

  useEffect(() => {
    try {
      (document as any).querySelector('div.ant-pro-global-header-logo > a > img').src = logo;
      (document as any).querySelector('div.ant-pro-global-header-logo > a > h1').innerText = title;
    } catch (error) {
      console.log(error);
    }
  }, []);

  return (
    <UmiSelectLang
      style={{
        padding: 4,
      }}
    />
  );
};

export const Question = () => {
  return (
    <div
      style={{
        display: 'flex',
        height: 26,
      }}
      onClick={() => {
        window.open('https://pro.ant.design/docs/getting-started');
      }}
    >
      <QuestionCircleOutlined />
    </div>
  );
};
