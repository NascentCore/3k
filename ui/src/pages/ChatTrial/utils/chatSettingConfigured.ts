import { generateUUID } from '.';

interface IConfigItem {
  id: string;
  modelType: string;
  apiKey: string;
  apiBase: string;
  apiModelName: string;
  apiContextLength: number;
  maxToken: number;
  chunkSize: number;
  context: number;
  temperature: number;
  top_P: number;
  top_K: number;
  capabilities: string[];
  active: boolean;
  modelName?: string;
}

const KEY = 'ChatSettingConfigured';
const defaultChatSettingConfigured = [
  {
    id: 'openAI',
    modelType: 'openAI',
    apiKey: '',
    apiBase: '',
    apiModelName: '',
    apiContextLength: 8192,
    maxToken: 512,
    chunkSize: 800,
    context: 0,
    temperature: 0.5,
    top_P: 1,
    top_K: 30,
    capabilities: ['mixedSearch'],
    active: true,
  },
  {
    id: 'ollama',
    modelType: 'ollama',
    apiKey: 'ollama',
    apiBase: '',
    apiModelName: '',
    apiContextLength: 2048,
    maxToken: 512,
    chunkSize: 800,
    context: 0,
    temperature: 0.5,
    top_P: 1,
    top_K: 30,
    capabilities: ['mixedSearch'],
    active: false,
  },
  {
    id: '',
    modelType: '自定义模型配置',
    apiKey: '',
    apiBase: '',
    apiModelName: '',
    apiContextLength: 8192,
    maxToken: 512,
    chunkSize: 800,
    context: 0,
    temperature: 0.5,
    top_P: 1,
    top_K: 30,
    capabilities: ['mixedSearch'],
    active: false,
    modelName: '',
  },
];

export const getChatSettingConfigured = (): IConfigItem[] => {
  const str = localStorage.getItem(KEY);
  if (str) {
    return JSON.parse(str);
  } else {
    localStorage.setItem(KEY, JSON.stringify(defaultChatSettingConfigured));
    return getChatSettingConfigured();
  }
};

export const getActiveChatSettingConfigured = () => {
  const list = getChatSettingConfigured();
  const activeItem = list.find((x: IConfigItem) => x.active);
  const params: any = {};
  if (activeItem) {
    params.max_token = activeItem.maxToken;
    params.api_base = activeItem.apiBase;
    params.api_key = activeItem.apiKey;
    params.model = activeItem.modelName;
    params.api_context_length = activeItem.apiContextLength;
    params.chunk_size = activeItem.chunkSize;
    params.top_p = activeItem.top_P;
    params.top_k = activeItem.top_K;
    params.temperature = activeItem.temperature;
    params.hybrid_search = activeItem.capabilities.includes('mixedSearch');
    params.networking = activeItem.capabilities.includes('networkSearch');
    params.only_need_search_results = activeItem.capabilities.includes('onlySearch');
    params.rerank = activeItem.capabilities.includes('rerank');
    params.using_model_knowledge = activeItem.capabilities.includes('llm');
    params.text_to_sql = activeItem.capabilities.includes('sql');
  }
  return params;
};

export const saveChatSettingConfigured = (data: IConfigItem): IConfigItem[] => {
  const list = getChatSettingConfigured();
  if (!data.id) {
    data.id = generateUUID();
  }
  if (data.active) {
    for (const item of list) {
      item.active = false;
    }
  }
  const hasIndex = list.findIndex((x: IConfigItem) => x.id === data.id);
  if (hasIndex === -1) {
    data.modelType = data.modelName || '';
    list.push(data);
  } else {
    list[hasIndex] = data;
  }
  localStorage.setItem(KEY, JSON.stringify(list));
  return list;
};
