export const saveToken = (token: string) => {
  localStorage.setItem('sxwl_token', token);
};

export const getToken = () => {
  return localStorage.getItem('sxwl_token');
};

export const removeToken = () => {
  localStorage.removeItem('sxwl_token');
};

export function formatFileSize(bytes: number): string {
  const megabytes = bytes / (1024 * 1024);
  const gigabytes = bytes / (1024 * 1024 * 1024);

  if (gigabytes >= 1) {
    return `${gigabytes.toFixed(2)} GB`;
  } else {
    return `${megabytes.toFixed(2)} MB`;
  }
}

export function copyTextToClipboard(text) {
  var textarea = document.createElement('textarea');
  textarea.value = text;
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  document.body.removeChild(textarea);
  console.log('Text copied to clipboard');
}

export const concatArray = (arr1: any[], arr2: any[]) => {
  const _arr1 = arr1 || [];
  const _arr2 = arr2 || [];
  return [..._arr1, ..._arr2];
};

interface WindowWithUser extends Window {
  __user?: {
    user_id?: string;
  };
}
export const removeUserIdPrefixFromPath = (str: string): string => {
  const windowWithUser = window as WindowWithUser;
  const user_id = windowWithUser.__user?.user_id;
  if (typeof user_id === 'string' && str.startsWith(user_id + '/')) {
    return str.substring(user_id.length + 1);
  }
  return str;
};
