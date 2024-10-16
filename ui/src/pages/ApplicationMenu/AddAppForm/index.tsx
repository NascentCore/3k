import { Button, Form, Input, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl, useModel } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';
import { apiPostAppJob, useApiGetInference } from '@/services';

interface IProps {
  onChange: () => void;
  onCancel: () => void;
  record: any;
}

const Index = ({ onChange, onCancel, record }: IProps) => {
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const intl = useIntl();
  const [inferenceOptions, setInferenceOptions] = useState([]);
  const { data: inferenceList, mutate, isLoading } = useApiGetInference();

  const [form] = Form.useForm();
  useEffect(() => {
    if (record) {
      form.setFieldsValue({});
    } else {
      form.setFieldsValue({});
    }
  }, [record]);

  useEffect(() => {
    form.setFieldsValue({});
    if (Array.isArray(inferenceList?.data)) {
      const options = inferenceList.data
      .filter((item: any) => item.status === 'running')
      .map((item: any) => ({
        label: `${item.model_name} (${item.service_name})`,
        value: item.api,
      }));
      setInferenceOptions(options);
    }
  }, [inferenceList]);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      const cleanedApiBase = values.api_addr ? values.api_addr.replace('/chat/completions', '') : '';
      const params = {
        app_id: record.app_id,
        app_name: record.app_name,
        instance_name: '',
        meta: JSON.stringify({env: {
          API_BASE: cleanedApiBase,
          }}),
      };
      return apiPostAppJob({ data: params }).then((res) => {
        message.success('Success');
        onChange();
      });
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          label={intl.formatMessage({
            id: 'pages.applicationMenu.appAddForm.form.app_name',
            defaultMessage: '应用名称',
          })}
        >
          {record.app_name}
        </Form.Item>
        <Form.Item
          label={intl.formatMessage({
            id: 'pages.applicationMenu.appAddForm.form.inference_select',
          })}
          name="api_addr"
          rules={[{ required: false, 
            message: intl.formatMessage({
              id: 'pages.applicationMenu.appAddForm.form.inference_select_placeholder',
            })
          }]}
        >
          <Select
            placeholder={intl.formatMessage({
              id: 'pages.applicationMenu.appAddForm.form.inference_select_placeholder',
            })}
            options={inferenceOptions}
          />
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.global.button.submit',
                defaultMessage: '提交',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.global.button.cancel',
                defaultMessage: '取消',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

export default Index;
