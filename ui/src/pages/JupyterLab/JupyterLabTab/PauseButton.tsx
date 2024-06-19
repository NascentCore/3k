import { apiPostJobJupyterlabPause } from '@/services';
import { useIntl } from '@umijs/max';
import { Button, Popconfirm, message } from 'antd';
import { useState } from 'react';

const Index = ({ mutate, record }: any) => {
  const intl = useIntl();
  const [disabled, setDisabled] = useState(false);
  return (
    <>
      <Popconfirm
        title={intl.formatMessage({ id: 'pages.global.confirm.title' })}
        onConfirm={() => {
          setDisabled(true);
          apiPostJobJupyterlabPause({ data: { job_name: record?.job_name } })
            .then(() => {
              mutate();
              message.success(
                intl.formatMessage({
                  id: 'pages.global.form.submit.success',
                  defaultMessage: '操作成功',
                }),
              );
            })
            .catch(() => {
              setDisabled(false);
            });
        }}
        okText={intl.formatMessage({ id: 'pages.global.confirm.okText' })}
        cancelText={intl.formatMessage({ id: 'pages.global.confirm.cancelText' })}
      >
        <Button type={'link'} disabled={disabled}>
          暂停
        </Button>
      </Popconfirm>
    </>
  );
};

export default Index;
