import { concatArray, removeUserIdPrefixFromPath } from '@/utils';
import { request, useIntl } from '@umijs/max';
import useSWR from 'swr';

const swrConfig = {
  revalidateIfStale: false,
  revalidateOnFocus: false,
  revalidateOnReconnect: false,
  shouldRetryOnError: false,
  refreshInterval: 1000 * 60 * 60,
};

// 登录接了 /auth/login
export async function apiAuthLogin(options?: { [key: string]: any }) {
  return request('/api/user/login', {
    method: 'POST',
    ...(options || {}),
  });
}

// 获取钉钉用户信息 /auth/login
export async function apiDingtalkUserInfo(options?: { [key: string]: any }) {
  return request('/api/user/login', {
    method: 'POST',
    ...(options || {}),
  });
}

export async function apiGetDingtalkUserInfo(code: string) {
  return request(`/api/dingtalk/userinfo?code=${code}`, {
    method: 'GET',
  });
}

// 获取用户信息 /auth/info
export async function apiAuthInfo(options?: { [key: string]: any }) {
  return request('/api/user/info', {
    method: 'GET',
    ...(options || {}),
  });
}

// 发送验证码
export async function apiCodeSendEmail(email: string) {
  return request('/api/user/email?email=' + email, {
    method: 'POST',
  });
}

// 注册
export async function apiUsersRegisterUser(codemes: string, options?: { [key: string]: any }) {
  return request('/api/user/register/' + codemes, {
    method: 'POST',
    ...(options || {}),
  });
}

// 模型列表
export async function apiResourceModels(options?: { [key: string]: any }) {
  return request('/api/resource/models', {
    method: 'GET',
    ...(options || {}),
  });
}

// 模型列表
export const useApiResourceModels = (options?: { [key: string]: any }) =>
  useSWR(
    ['/api/resource/models', options],
    ([url, data]) => {
      return request(url, {
        method: 'GET',
        ...(data || {}),
      });
    },
    swrConfig,
  );
export const useResourceModelsOptions = () => {
  const { data }: any = useApiResourceModels();
  const options = concatArray(data?.public_list, data?.user_list).map((item: any) => {
    return {
      ...item,
      label: removeUserIdPrefixFromPath(item.name),
      value: item.id,
      key: item.id,
    };
  });
  return options;
};

// 数据集列表
export async function apiResourceDatasets(options?: { [key: string]: any }) {
  return request('/api/resource/datasets', {
    method: 'GET',
    ...(options || {}),
  });
}
// 数据集列表
export const useApiResourceDatasets = (options?: { [key: string]: any }) =>
  useSWR(
    ['/api/resource/datasets', options],
    ([url, data]) => {
      return request(url, {
        method: 'GET',
        ...(data || {}),
      });
    },
    swrConfig,
  );

export const useResourceDatasetsOptions = () => {
  const { data }: any = useApiResourceDatasets();
  const options = concatArray(data?.public_list, data?.user_list).map((x) => ({
    ...x,
    label: removeUserIdPrefixFromPath(x.name),
    value: x.id,
    key: x.id,
  }));
  return options;
};

// 适配器 api/resource/adapters
export async function apiResourceAdapters(options?: { [key: string]: any }) {
  return request('/api/resource/adapters', {
    method: 'GET',
    ...(options || {}),
  });
}
export const useApiResourceAdapters = () =>
  useSWR(['/api/resource/adapters'], apiResourceAdapters, swrConfig);
export const useResourceAdaptersOptions = () => {
  const { data }: any = useApiResourceAdapters();
  const options = concatArray(data?.public_list, data?.user_list).map((x) => ({
    ...x,
    label: removeUserIdPrefixFromPath(x.name),
    value: x.id,
    key: x.id,
  }));
  return options;
};

// 2.6 无代码微调

export async function apiFinetunes(options?: { [key: string]: any }) {
  return request('/api/job/finetune', {
    method: 'POST',
    ...(options || {}),
  });
}

// 2.7.1 推理服务部署 /
export async function apiInference(options?: { [key: string]: any }) {
  return request('/api/job/inference', {
    method: 'POST',
    ...(options || {}),
  });
}

// 2.7.2 查询推理服务状态
export async function apiGetInference(options?: { [key: string]: any }) {
  return request('/api/job/inference', {
    method: 'GET',
    ...(options || {}),
  });
}

// 推理服务列表
export const useApiGetInference = (options?: { [key: string]: any }) =>
  useSWR(['/api/job/inference', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 推理服务删除
export async function apiDeleteInference(options?: { [key: string]: any }) {
  return request('/api/job/inference', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 查询任务详情 /api/userJob
export async function apiGetUserJob(options?: { [key: string]: any }) {
  return request('/api/job/training', {
    method: 'GET',
    ...(options || {}),
  });
}

// 任务提交  /api/userJob
export async function apiPostUserJob(options?: { [key: string]: any }) {
  return request('/api/job/training', {
    method: 'POST',
    ...(options || {}),
  });
}

// 任务列表
export const useApiGetUserJob = (options?: { [key: string]: any }) =>
  useSWR(['/api/job/training', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 删除任务
export async function apiDeleteUserJob(options?: { [key: string]: any }) {
  return request('/api/userJob/job_del', {
    method: 'POST',
    ...(options || {}),
  });
}

// GPU 列表查询
export const useApiGetGpuType = (options?: { [key: string]: any }) =>
  useSWR(
    ['/api/resource/gpus', options],
    ([url, data]) => {
      return request(url, {
        method: 'GET',
        ...(data || {}),
      });
    },
    swrConfig,
  );
export const useGpuTypeOptions = () => {
  const intl = useIntl();
  const { data }: any = useApiGetGpuType();
  const options =
    data?.map((x: any) => ({
      ...x,
      label: x.gpuProd,
      value: x.gpuProd,
    })) || [];
  return [
    ...options,
    {
      label: `A100 (${intl.formatMessage({
        id: 'pages.golbal.gpu.select.option.disabled.placeholder',
        defaultMessage: '充值超过 10000 可选',
      })})`,
      value: 'A100',
      disabled: true,
    },
    {
      label: `H100 (${intl.formatMessage({
        id: 'pages.golbal.gpu.select.option.disabled.placeholder',
        defaultMessage: '充值超过 10000 可选',
      })})`,
      value: 'H100',
      disabled: true,
    },
  ];
};

// 集群信息 新增
export async function apiPostApiNode(options?: { [key: string]: any }) {
  // return Promise.resolve();
  return request('/api/cluster/node', {
    method: 'POST',
    ...(options || {}),
  });
}

// 集群信息 列表
export const useGetApiNode = (options?: { [key: string]: any }) =>
  useSWR(['/api/cluster/node', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 用户配额 列表接口 /api/quota
export const useGetApiQuota = (options?: { [key: string]: any }) =>
  useSWR(['/api/resource/quota', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 用户配额新增接口 /quota
export async function apiPostQuota(options?: { [key: string]: any }) {
  return request('/api/resource/quota', {
    method: 'POST',
    ...(options || {}),
  });
}

// 修改配额
export async function apiPutQuota(options?: { [key: string]: any }) {
  return request('/api/resource/quota', {
    method: 'PUT',
    ...(options || {}),
  });
}

// 删除配额
export async function apiDeleteQuota(options?: { [key: string]: any }) {
  return request('/api/resource/quota', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 用户列表查询
export const useGetApiUser = (options?: { [key: string]: any }) =>
  useSWR(['/api/user/users', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 镜像列表
export async function apiGetJobJupyterImage(options?: { [key: string]: any }) {
  return request('/api/job/jupyter/image', {
    method: 'GET',
    ...(options || {}),
  });
}
export const useApiGetJobJupyterImage = (options?: { [key: string]: any }) =>
  useSWR(['/api/job/jupyter/image', options], ([url, options]) => {
    return apiGetJobJupyterImage(options);
  });

// 查询镜像版本
export async function apiGetJobJupyterImageversion(options?: { [key: string]: any }) {
  return request('/api/job/jupyter/imageversion', {
    method: 'GET',
    ...(options || {}),
  });
}

export const useApiGetJobJupyterImageversion = (options?: { [key: string]: any }) =>
  useSWR(['/api/job/jupyter/imageversion', options], ([url, options]) => {
    return apiGetJobJupyterImageversion(options);
  });

// 镜像列表 删除
export async function apiDeleteJobJupyterImage(options?: { [key: string]: any }) {
  return request('/api/job/jupyter/image', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 构建镜像
export async function apiPostJobJupyterImage(options?: { [key: string]: any }) {
  return request('/api/job/jupyter/image', {
    method: 'POST',
    ...(options || {}),
  });
}

// 查询jupyterlab实例列表
export async function apiGetJobJupyterlab(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab', {
    method: 'GET',
    ...(options || {}),
  });
}

export const useApiGetJobJupyterlab = (options?: { [key: string]: any }) =>
  useSWR(['/api/job/jupyterlab', options], ([url, options]) => {
    return apiGetJobJupyterlab(options);
  });

// 删除jupyterlab实例
export async function apiDeleteJobJupyterlab(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 创建jupyterlab实例
export async function apiPostJobJupyterlab(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab', {
    method: 'POST',
    ...(options || {}),
  });
}

// 修改jupyterlab实例
export async function apiPutJobJupyterlab(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab', {
    method: 'PUT',
    ...(options || {}),
  });
}

// jupyterlab 暂停
export async function apiPostJobJupyterlabPause(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab/pause', {
    method: 'POST',
    ...(options || {}),
  });
}

// jupyterlab 启动
export async function apiPostJobJupyterlabResume(options?: { [key: string]: any }) {
  return request('/api/job/jupyterlab/resume', {
    method: 'POST',
    ...(options || {}),
  });
}

// 基础镜像列表
export async function apiGetResourceBaseimages(options?: { [key: string]: any }) {
  return request('/api/resource/baseimages', {
    method: 'GET',
    ...(options || {}),
  });
}

export const useApiGetResourceBaseimages = (options?: { [key: string]: any }) =>
  useSWR(['/api/resource/baseimages', options], ([, options]) => {
    return apiGetResourceBaseimages(options);
  });

// 余额查询 /api/pay/balance?user_id=<long>
export async function apiGetPayBalance(options?: { [key: string]: any }) {
  return request('/api/pay/balance', {
    method: 'GET',
    ...(options || {}),
  });
}
export const useApiGetPayBalance = (options?: { [key: string]: any }) =>
  useSWR(['/api/pay/balance', options], ([, options]) => {
    return apiGetPayBalance(options);
  });

// 充值
export async function apiPostPayBalance(options?: { [key: string]: any }) {
  return request('/api/pay/balance', {
    method: 'POST',
    ...(options || {}),
  });
}
// 充值记录
export async function apiGetPayRecharge(options?: { [key: string]: any }) {
  return request('/api/pay/recharge', {
    method: 'GET',
    ...(options || {}),
  });
}
export const useApiGetPayRecharge = (options?: { [key: string]: any }) =>
  useSWR(['/api/pay/recharge', options], ([, options]) => {
    return apiGetPayRecharge(options);
  });

// 账单查询 /api/pay/billing?user_id=
// {{baseUrl}}/api/pay/billing?user_id=<long>&start_time=<string>&end_time=<string>&job_id=<string>
export async function apiGetPayBilling(options?: { [key: string]: any }) {
  return request('/api/pay/billing', {
    method: 'GET',
    ...(options || {}),
  });
}

export const useApiGetPayBilling = (options?: { [key: string]: any }) =>
  useSWR(['/api/pay/billing', options], ([, options]) => {
    return apiGetPayBilling(options);
  });

// 查询用户账单
export async function apiGetPayBillingTasks(options?: { [key: string]: any }) {
  return request('/api/pay/billing/tasks', {
    method: 'GET',
    ...(options || {}),
  });
}
export const useApiGetPayBillingTasks = (options?: { [key: string]: any }) =>
  useSWR(['/api/pay/billing/tasks', options], ([, options]) => {
    return apiGetPayBillingTasks(options);
  });
export async function apiClusterCpods(options?: { [key: string]: any }) {
  return request('/api/cluster/cpods', {
    method: 'GET',
    ...(options || {}),
  }).then((data) => {
    const groupedData = data.data.reduce((grouped: any, item: any) => {
      const cpodId = item.cpod_id;
      if (!grouped[cpodId]) {
        grouped[cpodId] = [];
      }
      grouped[cpodId].push(item);
      return grouped;
    }, {});
    return groupedData;
  });
}
export const useApiClusterCpods = (options?: { [key: string]: any }) =>
  useSWR(['/api/cluster/cpods', options], ([, options]) => {
    return apiClusterCpods(options);
  });

// 查询应用
export async function apiGetAppJob() {
  return request('/api/app/job', {
    method: 'GET',
  });
}

// 创建应用
export async function apiPostAppJob(options?: { [key: string]: any }) {
  return request('/api/app/job', {
    method: 'POST',
    ...(options || {}),
  });
}

// 终止应用
export async function apiDeleteAppJob(options?: { [key: string]: any }) {
  return request('/api/app/job', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 应用注册列表
export async function apiGetAppList() {
  return request('/api/app/list', {
    method: 'GET',
  });
}

// 应用注册接口
export async function apiPostAppRegister(options?: { [key: string]: any }) {
  return request('/api/app/register', {
    method: 'POST',
    ...(options || {}),
  });
}

// 删除
export async function apiDeleteAppList(options?: { [key: string]: any }) {
  return request('/api/app/delete', {
    method: 'DELETE',
    ...(options || {}),
  });
}
