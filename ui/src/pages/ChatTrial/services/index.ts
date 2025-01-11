import { request } from '@umijs/max';
import useSWR from 'swr';

export const BaseUrl = '/';
export async function api_get_file_base64(data: { file_id: string; user_id: string }) {
  return request(`${BaseUrl}/api/local_doc_qa/get_file_base64`, {
    method: 'POST',
    data,
  });
}

// 查询知识库列表
export async function api_list_knowledge_base() {
  return request(`${BaseUrl}/api/local_doc_qa/list_knowledge_base`, {
    method: 'POST',
    data: {
      user_id: 'zzp',
    },
  });
}

// 查询知识库文档 ${BaseUrl}/api/local_doc_qa/list_files
export async function api_list_files(params: {
  kb_id: string;
  page_id: number;
  page_limit: number;
  user_id: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/list_files`, {
    method: 'POST',
    data: params,
  });
}

export const use_api_list_files = (params: {
  kb_id: string;
  page_id: number;
  page_limit: number;
  user_id: string;
}) =>
  useSWR([`${BaseUrl}/api/local_doc_qa/list_files`, params], ([, params]) => {
    return api_list_files(params).then((res) => res.data);
  });

// 新增知识库
export async function api_new_knowledge_base(params: {
  kb_id?: string;
  kb_name: string;
  user_id: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/new_knowledge_base`, {
    method: 'POST',
    data: params,
  });
}

// 删除知识库
export async function api_delete_knowledge_base(params: { kb_ids: string[]; user_id: string }) {
  return request(`${BaseUrl}/api/local_doc_qa/delete_knowledge_base`, {
    method: 'POST',
    data: params,
  });
}

// 重命名
export async function api_rename_knowledge_base(params: {
  kb_id: string;
  new_kb_name: string;
  user_id: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/rename_knowledge_base`, {
    method: 'POST',
    data: params,
  });
}

// 添加网址
export async function api_upload_weblink(params: {
  chunk_size: number;
  kb_id: string;
  mode: string;
  url: string;
  user_id: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/upload_weblink`, {
    method: 'POST',
    data: params,
  });
}

// 录入问答 ${BaseUrl}/api/local_doc_qa/upload_faqs
export async function api_upload_faqs(params: {
  user_id: string;
  kb_id: string;
  faqs: { question: string; answer: string; nos_key: null }[];
  chunk_size: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/upload_faqs`, {
    method: 'POST',
    data: params,
  });
}

// 删除问答

export async function api_delete_files(params: {
  user_id: string;
  kb_id: string;
  file_ids: string[];
}) {
  return request(`${BaseUrl}/api/local_doc_qa/delete_files`, {
    method: 'POST',
    data: params,
  });
}

export async function api_get_doc_completed(params: {
  file_id: string;
  kb_id: string;
  page_id: number;
  page_limit: number;
  user_id: string;
}) {
  return request(`${BaseUrl}/api/local_doc_qa/get_doc_completed`, {
    method: 'POST',
    data: params,
  });
}

export const use_api_get_doc_completed = (params: {
  show: boolean;
  file_id: string;
  kb_id: string;
  page_id: number;
  page_limit: number;
  user_id: string;
}) =>
  useSWR(
    [`${BaseUrl}/api/local_doc_qa/get_doc_completed`, params],
    params.show
      ? ([, params]) => {
          return api_get_doc_completed(params);
        }
      : null,
  );
