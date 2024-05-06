import { request } from '@umijs/max';
import useSWR from 'swr';

// 登录接了 /auth/login
export async function apiAuthLogin(options?: { [key: string]: any }) {
  return request('/api/user/login', {
    method: 'POST',
    ...(options || {}),
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
  useSWR(['/api/resource/models', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

// 数据集列表
export async function apiResourceDatasets(options?: { [key: string]: any }) {
  return request('/api/resource/datasets', {
    method: 'GET',
    ...(options || {}),
  });
}
// 数据集列表
export const useApiResourceDatasets = (options?: { [key: string]: any }) =>
  useSWR(['/api/resource/datasets', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

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
  useSWR(['/api/resource/gpus', options], ([url, data]) => {
    return request(url, {
      method: 'GET',
      ...(data || {}),
    });
  });

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

// 基础镜像列表
export async function apiGetResourceBaseimages(options?: { [key: string]: any }) {
  return request('/api/resource/baseimages', {
    method: 'GET',
    ...(options || {}),
  });
}

export const useApiGetResourceBaseimages = (options?: { [key: string]: any }) =>
  useSWR([options], ([options]) => {
    return apiGetResourceBaseimages(options);
  });
