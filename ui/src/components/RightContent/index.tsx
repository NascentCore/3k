import { QuestionCircleOutlined } from '@ant-design/icons';
import { SelectLang as UmiSelectLang } from '@umijs/max';
import React, { useEffect } from 'react';
import { useIntl } from '@umijs/max';

export type SiderTheme = 'light' | 'dark';

export const SelectLang = () => {
  const intl = useIntl();

  const title = intl.formatMessage({
    id: 'app.title',
    defaultMessage: '中科苏州',
  });

  useEffect(() => {
    try {
      (document as any).querySelector('div.ant-pro-global-header-logo').innerHTML = `
      <img style="margin-left: 10px" src="/icons/zksz.jpg"></img>
       `;
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
