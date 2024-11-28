export async function getPlaygroundModels() {
  return request('/api/job/inference/playground', {
    method: 'GET',
  });
} 