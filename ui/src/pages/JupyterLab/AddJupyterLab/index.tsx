import {
  apiPostJobJupyterlab,
  apiPutJobJupyterlab,
  useGpuTypeOptions,
  useResourceAdaptersOptions,
  useResourceDatasetsOptions,
  useResourceModelsOptions,
} from '@/services';
import { Button, Form, Input, Select, message } from 'antd';
import { useEffect } from 'react';
import { useIntl, useModel } from '@umijs/max';
import AsyncButton from '@/components/AsyncButton';

interface IProps {
  onChange: () => void;
  onCancel: () => void;
  addJupterLabType: any;
  addJupterLabRecord: any;
}

const Index = ({ addJupterLabType, addJupterLabRecord, onChange, onCancel }: IProps) => {
  const { initialState } = useModel('@@initialState');
  const { currentUser } = initialState || {};
  const intl = useIntl();

  const gpuTypeOptions = useGpuTypeOptions();
  const modelsOptions = useResourceModelsOptions();
  const datasetsOptions = useResourceDatasetsOptions();
  const adaptersOptions = useResourceAdaptersOptions();

  const editDisabled = addJupterLabType === 'edit';

  const [form] = Form.useForm();
  useEffect(() => {
    if (addJupterLabRecord) {
      form.setFieldsValue({
        instance_name: addJupterLabRecord.instance_name,
        cpu_count: addJupterLabRecord.cpu_count,
        memory: addJupterLabRecord.memory / 1024 / 1024,
        gpu_count: addJupterLabRecord.gpu_count,
        gpu_product: addJupterLabRecord.gpu_product,
        data_volume_size: addJupterLabRecord.data_volume_size / 1024 / 1024,
        resource: {
          models: addJupterLabRecord.resource?.models?.map((x) => x.model_id) || [],
          datasets: addJupterLabRecord.resource?.datasets?.map((x) => x.dataset_id) || [],
          adapters: addJupterLabRecord.resource?.adapters?.map((x) => x.adapter_id) || [],
        },
      });
    } else {
      form.setFieldsValue({
        cpu_count: 2,
        memory: 2048,
        data_volume_size: 1024,
      });
    }
  }, [addJupterLabRecord]);
  const gpu_product_watch = Form.useWatch('gpu_product', form);

  const onFinish = () => {
    return form.validateFields().then(() => {
      const values = form.getFieldsValue();
      if (addJupterLabType === 'add') {
        console.log('Form values:', values);
        const models = modelsOptions
          .filter((x) => values.resource.models?.includes(x.value))
          ?.map((x) => ({
            model_id: x.id,
            model_name: x.name,
            model_size: x.size,
            model_is_public: x.is_public,
            model_template: x.template,
            model_path: '/models',
          }));
        const datasets = datasetsOptions
          .filter((x) => values.resource.datasets?.includes(x.value))
          ?.map((x) => ({
            dataset_id: x.id,
            dataset_name: x.name,
            dataset_size: x.size,
            dataset_is_public: x.is_public,
            dataset_path: '/datasets',
          }));
        const adapters = adaptersOptions
          .filter((x) => values.resource.adapters?.includes(x.value))
          ?.map((x) => ({
            adapter_id: x.id,
            adapter_name: x.name,
            adapter_size: x.size,
            adapter_is_public: x.is_public,
            adapter_path: '/adapters',
          }));
        const params = {
          ...values,
          user_id: currentUser?.user_id,
          cpu_count: Number(values.cpu_count),
          memory: Number(values.memory) * 1024 * 1024,
          data_volume_size: Number(values.data_volume_size) * 1024 * 1024,
          gpu_count: values.gpu_count ? Number(values.gpu_count) : void 0,
          resource: {
            models,
            datasets,
            adapters,
          },
        };
        console.log('Form submit:', params);
        return apiPostJobJupyterlab({
          data: params,
        }).then(() => {
          onChange();
          message.success(
            intl.formatMessage({
              id: 'pages.global.form.submit.success',
              defaultMessage: '添加成功',
            }),
          );
          onCancel();
        });
      }
      if (addJupterLabType === 'edit') {
        console.log('Form values:', values);
        const params = {
          job_name: addJupterLabRecord.job_name,
          gpu_product: values.gpu_product,
          cpu_count: Number(values.cpu_count),
          memory: Number(values.memory) * 1024 * 1024,
          data_volume_size: Number(values.data_volume_size) * 1024 * 1024,
          gpu_count: values.gpu_count ? Number(values.gpu_count) : void 0,
        };
        console.log('Form update:', params);
        return apiPutJobJupyterlab({ data: params }).then(() => {
          onChange();
          message.success(
            intl.formatMessage({
              id: 'pages.global.form.submit.success',
              defaultMessage: '操作成功',
            }),
          );
          onCancel();
        });
      }
    });
  };

  return (
    <>
      <Form form={form} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} style={{ maxWidth: 600 }}>
        <Form.Item
          name="instance_name"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.instance_name',
            defaultMessage: '实例名称',
          })}
          rules={[
            {
              required: true,
            },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message: intl.formatMessage({
                id: 'pages.jupyterLab.AddJupyterLab.form.instance_name.pattern',
                defaultMessage: '实例名称只能包含大小写字母、数字、下划线(_)和分割符(-)',
              }),
            },
          ]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            disabled={editDisabled}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="cpu_count"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.cpu_count',
            defaultMessage: 'CPU',
          })}
          rules={[{ required: true }]}
        >
          <Input
            type="text"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="memory"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.memory',
            defaultMessage: 'memory',
          })}
          rules={[{ required: true }]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            suffix="MB"
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="gpu_count"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.gpu_count',
            defaultMessage: 'GPU数量',
          })}
          rules={gpu_product_watch && [{ required: true }]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            min={1}
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="gpu_product"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.gpu_product',
            defaultMessage: 'GPU类型',
          })}
        >
          <Select
            allowClear
            options={gpuTypeOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
          />
        </Form.Item>
        <Form.Item
          name="data_volume_size"
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.data_volume_size',
            defaultMessage: '数据卷大小',
          })}
          rules={[{ required: true }]}
        >
          <Input
            type="number"
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            allowClear
            suffix="MB"
          />
        </Form.Item>
        <Form.Item
          name={['resource', 'models']}
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.model_id',
            defaultMessage: '挂载模型',
          })}
          extra={`${intl.formatMessage({
            id: 'pages.UserJobCommit.form.pretrainedModelPath',
          })}: /models`}
        >
          <Select
            allowClear
            mode="multiple"
            options={modelsOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            disabled={editDisabled}
          />
        </Form.Item>
        <Form.Item
          name={['resource', 'datasets']}
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.datasets',
            defaultMessage: '挂载数据集',
          })}
          extra={`${intl.formatMessage({
            id: 'pages.UserJobCommit.form.pretrainedModelPath',
          })}: /datasets`}
        >
          <Select
            allowClear
            mode="multiple"
            options={datasetsOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            disabled={editDisabled}
          />
        </Form.Item>

        <Form.Item
          name={['resource', 'adapters']}
          label={intl.formatMessage({
            id: 'pages.jupyterLab.AddJupyterLab.form.adapters',
            defaultMessage: '挂载适配器',
          })}
          extra={`${intl.formatMessage({
            id: 'pages.UserJobCommit.form.pretrainedModelPath',
          })}: /adapters`}
        >
          <Select
            allowClear
            mode="multiple"
            options={adaptersOptions}
            placeholder={intl.formatMessage({
              id: 'pages.global.form.placeholder',
              defaultMessage: '请输入',
            })}
            disabled={editDisabled}
          />
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
