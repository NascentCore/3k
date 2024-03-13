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