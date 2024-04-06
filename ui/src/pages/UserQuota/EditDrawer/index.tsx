/**
 * @name 微调
 * @description 微调
 */
import { apiFinetunes, apiResourceDatasets, useApiGetGpuType } from '@/services';
import { Button, Col, Drawer, Form, Input, Row, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { history } from '@umijs/max';
import { useIntl } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

const Content = ({ type, record, onCancel }) => {
  const intl = useIntl();

  const [form] = Form.useForm();
  const [formValues, setFormValues] = useState({});

  const [resourceDatasetsOption, setResourceDatasets] = useState([]);
  useEffect(() => {
    apiResourceDatasets({}).then((res) => {
      setResourceDatasets(res?.map((x) => ({ ...x, label: x.name, value: x.name })));
    });
  }, []);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      setFormValues(values);
      console.log('Form values:', values);
      return Promise.resolve({
        data: { ...values },
      }).then((res) => {
        message.success(
          intl.formatMessage({
            id: 'pages.userQuota.edit.form.success',
            defaultMessage: '操作成功',
          }),
        );
        onCancel();
      });
    });
  };

  return (
    <>
      <Form
        form={form}
        initialValues={formValues}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        style={{ maxWidth: 600 }}
      >
        <Form.Item
          name="name"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.name',
            defaultMessage: '用户',
          })}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.name',
              defaultMessage: '用户',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="node_role"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.role',
            defaultMessage: '资源类型',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Select
            allowClear
            options={[
              {
                label: 'worker',
                value: 'worker',
              },
            ]}
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.role',
              defaultMessage: '资源类型',
            })}
          />
        </Form.Item>

        <Form.Item
          name="gpu_product"
          label={intl.formatMessage({
            id: 'pages.userQuota.edit.form.gpu_product',
            defaultMessage: '资源配额',
          })}
          rules={[
            {
              required: true,
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.userQuota.edit.form.gpu_product',
              defaultMessage: '资源配额',
            })}
            allowClear
          />
        </Form.Item>

        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <AsyncButton type="primary" block onClick={onFinish}>
              {intl.formatMessage({
                id: 'pages.userQuota.edit.form.confirm',
                defaultMessage: '确定',
              })}
            </AsyncButton>
            <Button type="default" onClick={() => onCancel()} block>
              {intl.formatMessage({
                id: 'pages.userQuota.edit.form.cancel',
                defaultMessage: '放弃',
              })}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </>
  );
};

const Index = ({ type, record }) => {
  const intl = useIntl();

  const [open, setOpen] = useState(false);

  const btnTitle =
    type === 'edit'
      ? intl.formatMessage({
          id: 'pages.userQuota.edit.form.title',
          defaultMessage: '修改配额',
        })
      : intl.formatMessage({
          id: 'pages.userQuota.add.form.title',
          defaultMessage: '添加配额',
        });

  return (
    <>
      <Button
        type={type === 'edit' ? 'link' : 'primary'}
        onClick={() => {
          setOpen(true);
        }}
      >
        {btnTitle}
      </Button>
      <Drawer
        width={1000}
        title={btnTitle}
        placement="right"
        onClose={() => setOpen(false)}
        open={open}
      >
        {open && <Content type={type} record={record} onCancel={() => setOpen(false)} />}
      </Drawer>
    </>
  );
};

export default Index;
