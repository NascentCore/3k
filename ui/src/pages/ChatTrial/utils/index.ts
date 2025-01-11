import { IChatItemMsg } from '@/models/chat-h5-model';
import { getActiveChatSettingConfigured } from './chatSettingConfigured';
import { BaseUrl } from '../services';
import { getToken } from '@/utils';

export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function scrollChatBodyToBottom(key: string, animate: boolean = true) {
  document.querySelector(`.messageContainer_${key}`)?.scrollTo({
    top: 9999999999,
    behavior: animate ? 'smooth' : void 0,
  });
}

export function scrollH5ChatBodyToBottom(animate: boolean = true) {
  document.querySelector(`#chat-container-h5`)?.scrollTo({
    top: 9999999999,
    behavior: animate ? 'smooth' : void 0,
  });
}

export const getChatResponseJsonFromResponseText = (responseText: string) => {
  let json: any = {};
  const lines = responseText.split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const data = JSON.parse(line.slice(6));
        if (data.msg === 'success stream chat') {
          json = data;
          break;
        }
      } catch (e) {
        console.error('解析JSON时出错:', e);
      }
    }
  }
  const { response, source_documents, show_images } = json;
  const msgId = generateUUID();
  const chatMsgItem: IChatItemMsg = {
    content: response,
    source_documents,
    show_images,
    role: 'assistant',
    id: msgId,
    date: '',
    raw: json,
  };
  return chatMsgItem;
};

export const getChatResponseJsonFromResponseText_modalType = (responseText: string) => {
  const msgId = generateUUID();
  const chatMsgItem: IChatItemMsg = {
    content: responseText,
    source_documents: [],
    show_images: [],
    role: 'assistant',
    id: msgId,
    date: '',
    raw: {},
  };
  return chatMsgItem;
};

export const cahtAction = async ({
  id,
  knowledgeListSelect,
  question,
  onMessage,
  onSuccess,
}: {
  id: string; // msgid
  question: string;
  knowledgeListSelect: string[];
  onMessage: (chatMsgItem: IChatItemMsg) => void;
  onSuccess: (chatMsgItem: IChatItemMsg) => void;
}) => {
  const condigParams = getActiveChatSettingConfigured();
  const response = await fetch(`${BaseUrl}/api/local_doc_qa/local_doc_chat`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream,application/json, text/event-stream',
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,mt;q=0.6,pl;q=0.5',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: 'zzp',
      kb_ids: knowledgeListSelect,
      history: [],
      question: question,
      product_source: 'saas',
      streaming: true,
      ...condigParams,
    }),
  });

  const reader = (response as any).body.getReader();
  const decoder = new TextDecoder();
  let fullChunk = '';
  let fullResponse = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    fullChunk += chunk;
    const lines = chunk.split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.response) {
            fullResponse += data.response;
            console.log('fullResponse', fullResponse);
            const chatMsgItem: IChatItemMsg = {
              content: fullResponse,
              role: 'assistant',
              id: id,
              date: '',
            };
            onMessage(chatMsgItem);
          }
        } catch (e) {
          console.error('解析JSON时出错:', e);
        }
      }
    }
  }
  const chatMsgItem2: IChatItemMsg = getChatResponseJsonFromResponseText(fullChunk);
  onSuccess({ ...chatMsgItem2, id: id });
};

export const cahtActionH5 = async ({
  id,
  knowledgeListSelect,
  question,
  onMessage,
  onSuccess,
}: {
  id: string; // msgid
  question: string;
  knowledgeListSelect: string[];
  onMessage: (chatMsgItem: IChatItemMsg) => void;
  onSuccess: (chatMsgItem: IChatItemMsg) => void;
}) => {
  const response = await fetch(`${BaseUrl}/api/local_doc_qa/local_doc_chat`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream,application/json, text/event-stream',
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,mt;q=0.6,pl;q=0.5',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      user_id: 'zzp',
      kb_ids: knowledgeListSelect,
      history: [],
      question: question.includes('产品图片')
        ? '产品图片'
        : question.includes('brochure')
        ? 'do you have the detailed information'
        : question.includes('宣传手册')
        ? '有详细资料吗'
        : question,
      streaming: true,
      networking: false,
      product_source: 'saas',
      rerank: false,
      only_need_search_results: false,
      hybrid_search: true,
      max_token: 512,
      api_base: 'https://ark.cn-beijing.volces.com/api/v3/',
      api_key: 'ff9ed2dd-cdf0-40d4-b4ec-d3aa19e2bd0b',
      model: 'ep-20240721110948-mdv29',
      api_context_length: 14336,
      chunk_size: 800,
      top_p: 1,
      top_k: 30,
      temperature: 0.01,
    }),
  });

  const reader = (response as any).body.getReader();
  const decoder = new TextDecoder();
  let fullChunk = '';
  let fullResponse = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    fullChunk += chunk;
    const lines = chunk.split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.response) {
            fullResponse += data.response;
            console.log('fullResponse', fullResponse);
            const chatMsgItem: IChatItemMsg = {
              content: fullResponse,
              role: 'assistant',
              id: id,
              date: '',
            };
            onMessage(chatMsgItem);
          }
        } catch (e) {
          console.error('解析JSON时出错:', e);
        }
      }
    }
  }
  const chatMsgItem2: IChatItemMsg = getChatResponseJsonFromResponseText(fullChunk);
  onSuccess({ ...chatMsgItem2, id: id });
};

export const cahtActionH5_ModalType = async ({
  id,
  question,
  onMessage,
  onSuccess,
}: {
  id: string; // msgid
  question: string;
  onMessage: (chatMsgItem: IChatItemMsg) => void;
  onSuccess: (chatMsgItem: IChatItemMsg) => void;
}) => {
  const response = await fetch(`/api/v1/chat/completions`, {
    method: 'POST',
    headers: {
      Accept: 'text/event-stream,application/json, text/event-stream',
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,mt;q=0.6,pl;q=0.5',
      'Content-Type': 'application/json',
      Authorization: getToken(),
      //'Bearer eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0YjgxMDczMmIxNWY0NzYxOTYyMGM4YmEyMDFiNTMwNSIsInN1YiI6InBsYXlncm91bmRAc3h3bC5haSIsInVzZXJfaWQiOiJ1c2VyLTc4ZGM1NTU3LTZiYjktNGMwZi05ZGQzLTIxZmE5YTc3MTM0OSIsInVzZXJpZCI6ODgsInVzZXJuYW1lIjoicGxheWdyb3VuZEBzeHdsLmFpIn0.RSyYZNQMH9LGrzo2qrDCwSNW97-8pEPi9fuAsU2SLzXRhD5Y5bNki8yCdHYG_WrfT1TR5bm3QO_gKBaX332xgQ',
    },
    body: JSON.stringify({
      stream: true,
      model: sessionStorage.getItem('chat-h5-model-model'), //'google/gemma-2b-it',
      messages: [
        {
          role: 'user',
          content: question,
        },
      ],
    }),
  });

  const reader = (response as any).body.getReader();
  const decoder = new TextDecoder();
  let fullChunk = '';
  let fullResponse = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    fullChunk += chunk;
    // console.log('chunk', chunk);
    const lines = chunk.split('\n');
    for (const line of lines) {
      if (line === 'data: [DONE]') {
        break;
      }
      if (line.startsWith('data: ')) {
        // console.log('line', line);
        try {
          const data = JSON.parse(line.slice(6));
          const content = data.choices[0].delta.content;
          if (content) {
            fullResponse += content;
            console.log('fullResponse', fullResponse);
            const chatMsgItem: IChatItemMsg = {
              content: fullResponse,
              role: 'assistant',
              id: id,
              date: '',
            };
            // console.log(Date.now())
            // console.log('chatMsgItem', chatMsgItem);
            onMessage(chatMsgItem);
          }
        } catch (e: any) {
          console.log('解析JSON时出错:', e.message);
        }
      }
    }
  }
  const chatMsgItem2: IChatItemMsg = getChatResponseJsonFromResponseText_modalType(fullResponse);
  // console.log('chatMsgItem2', chatMsgItem2);
  onSuccess({ ...chatMsgItem2, id: id });
};
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    // 对于小于1KB的文件大小，直接返回字节
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    // 对于小于1MB的文件大小，转换为KB
    const kilobytes = bytes / 1024;
    return `${kilobytes.toFixed(2)} KB`;
  } else if (bytes < 1024 * 1024 * 1024) {
    // 对于小于1GB的文件大小，转换为MB
    const megabytes = bytes / (1024 * 1024);
    return `${megabytes.toFixed(2)} MB`;
  } else {
    // 对于大于等于1GB的文件大小，转换为GB
    const gigabytes = bytes / (1024 * 1024 * 1024);
    return `${gigabytes.toFixed(2)} GB`;
  }
}

export function formatTimestamp(timestamp: string): string {
  if (timestamp.length !== 12) {
    throw new Error('Invalid timestamp length. Expected length is 12.');
  }
  // 提取年、月、日、小时和分钟
  const year = timestamp.substring(0, 4);
  const month = timestamp.substring(4, 6);
  const day = timestamp.substring(6, 8);
  const hour = timestamp.substring(8, 10);
  const minute = timestamp.substring(10, 12);

  // 返回格式化的时间字符串
  return `${year}-${month}-${day} ${hour}:${minute}`;
}

export function detectDeviceType() {
  const isMobile = window.innerWidth <= 768; // 768px通常被认为是移动设备和PC的分界线
  if (isMobile) {
    return 'mobile';
  } else {
    return 'pc';
  }
}

// 文件扩展名映射到对应的data URI scheme
export const FileMimeTypeMap: any = {
  md: 'text/markdown',
  txt: 'text/plain',
  pdf: 'application/pdf',
  jpg: 'image/jpeg',
  png: 'image/png',
  jpeg: 'image/jpeg',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  eml: 'message/rfc822',
  csv: 'text/csv',
};

// 获取文件后缀
export function getFileExtension(filename: string): any {
  const lastDotIndex = filename.lastIndexOf('.');
  if (lastDotIndex === -1 || lastDotIndex === 0) {
    return '';
  }
  return filename.substring(lastDotIndex + 1);
}

export function base64ToBlobUrl(base64Data: string, mimeType: string) {
  const byteCharacters = atob(base64Data);
  const byteArrays = [];
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  const blob = new Blob(byteArrays, { type: mimeType });
  const blobUrl = URL.createObjectURL(blob);
  return blobUrl;
}

export function filterSourceDocuments(source_documents: any) {
  const source_documents_map: any = {};

  for (const source_documents_item of source_documents) {
    const { file_id, file_name, retrieval_query } = source_documents_item;

    // 提取 query 中所有 k/K 加数字 或 m/M 加数字（忽略大小写）
    const queryPattern = /(k\d+|m\d+)/gi; // 全局匹配，忽略大小写
    const queryMatches = retrieval_query.match(queryPattern);
    const queryMatchValues = queryMatches ? queryMatches.map((v) => v.toLowerCase()) : [];

    // 如果提取到了多个 k/M 数字，文件名中包含任意一项即可
    if (queryMatchValues.length > 0) {
      const fileNameLower = file_name ? file_name.toLowerCase() : '';
      const anyMatchIncluded = queryMatchValues.some((match) => fileNameLower.includes(match));
      if (!anyMatchIncluded) {
        continue; // 如果文件名不包含任意一个匹配项，跳过
      }
    }

    if (retrieval_query.includes('联系方式') || retrieval_query.includes('contact information')) {
      const isTure = file_name && file_name.includes('产品介绍');
      if (!isTure) {
        continue;
      }
    }

    if (retrieval_query.includes('哪些产品')) {
      const isTure = file_name && /\.(jpg|jpeg)$/.test(file_name.toLowerCase());
      if (!isTure) {
        continue;
      }
    }

    if (retrieval_query.includes('产品图片') || retrieval_query.includes('product pictures')) {
      const isTure = file_name && /\.(pdf)$/.test(file_name.toLowerCase());
      if (!isTure) {
        continue;
      }
    }

    if (retrieval_query.includes('详细资料') || retrieval_query.includes('宣传手册')) {
      const isTure = file_name && /\.(xls|xlsx)$/.test(file_name.toLowerCase());
      if (!isTure) {
        continue;
      }
    }

    if (!source_documents_map[file_id]) {
      source_documents_map[file_id] = source_documents_item;
    }
  }
  return Object.values(source_documents_map);
}

export function getParameterByName(name: string, url?: string) {
  let myUrl: any = url;
  if (!url) myUrl = window.location.href;
  const regex = new RegExp(`[?&]${name}(=([^&#]*)|&|#|$)`);
  const results = regex.exec(myUrl);
  if (!results) return null;
  if (!results[2]) return '';
  try {
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
  } catch (e) {
    return '';
  }
}
