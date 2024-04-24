import { Button, Form, Select, message } from 'antd';
import { useEffect } from 'react';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';
import { apiGetResourceBaseimages, apiPostJobJupyterImage, useApiResourceModels } from '@/services';

interface IProps {
  record: any;
  onChange: () => void;
  onCancel: () => void;
}

const Index = ({ record, onChange, onCancel }: IProps) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  useEffect(() => {
    form.setFieldsValue({});
    apiGetResourceBaseimages();
  }, []);

  const { data: resourceModels }: any = useApiResourceModels();
  const resourceModelsList = resourceModels?.map((x) => ({
    ...x,
    label: (
      <>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12,
          }}
        >
          <span style={{ marginRight: 20 }}>{x.name}</span>
          {/* <span>{formatFileSize(x.size)}</span> */}
        </div>
      </>
    ),
    value: x.id,
    key: x.id,
  }));

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      console.log('Form values:', values);
      return apiPostJobJupyterImage({
        data: {
          base_image: values.base_image,
          instance_name: record.instance_name,
        },
      }).then(() => {
        onChange();
        message.success(
          intl.formatMessage({
            id: 'pages.global.form.submit.success',
            defaultMessage: '操作成功',
          }),
        );
        onCancel();
      });
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="base_image"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.JupyterLabTab.BuildingImage.form.base_image',
            defaultMessage: '基座镜像',
          })}
          rules={[{ required: true }]}
        >
          <Select
            allowClear
            options={resourceModelsList}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.select.placeholder',
              defaultMessage: '请选择',
            })}
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          {intl
            .formatMessage({
              id: 'pages.jupyterLab.JupyterLabTab.BuildingImage.form.tips',
            })
            ?.split('<br/>')
            .map((x) => (
              <p>{x}</p>
            ))}
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.confirm',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.clusterInformation.add.form.cancel',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

export default Index;
