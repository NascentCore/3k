import { request } from '@umijs/max';

// /auth/login
export async function apiAuthLogin(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/auth/login', {
    method: 'POST',
    ...(options || {}),
  });
}

// /auth/info
export async function apiAuthInfo(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/auth/info', {
    method: 'GET',
    ...(options || {}),
  });
}

// 2.1 查询模型和数据集
export async function apiResourceModels(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/resource/models', {
    method: 'GET',
    ...(options || {}),
  });
}

export async function apiResourceDatasets(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/resource/datasets', {
    method: 'GET',
    ...(options || {}),
  });
}

// 2.6 无代码微调

export async function apiFinetunes(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/finetune', {
    method: 'POST',
    ...(options || {}),
  });
}

// 2.7.1 推理服务部署 /
export async function apiInference(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/inference', {
    method: 'POST',
    ...(options || {}),
  });
}

// 2.7.2 查询推理服务状态
export async function apiGetInference(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/inference', {
    method: 'GET',
    ...(options || {}),
  });
}

export async function apiDeleteInference(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/inference', {
    method: 'DELETE',
    ...(options || {}),
  });
}

// 查询任务详情 /api/userJob
export async function apiGetUserJob(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/userJob', {
    method: 'GET',
    ...(options || {}),
  });
}

// 删除任务
export async function apiDeleteUserJob(options?: { [key: string]: any }) {
  return request<API.NoticeIconList>('/api/userJob', {
    method: 'DELETE',
    ...(options || {}),
  });
}
