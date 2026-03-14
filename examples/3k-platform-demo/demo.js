/**
 * 3K Platform Demo - Mock data and navigation
 * This demo showcases platform functionality without backend/API calls.
 */

const MOCK_MODELS = [
  { name: 'LLaMA2-7B', owner: 'meta-llama', category: 'LLM', size: '13GB', tags: ['finetune', 'inference'] },
  { name: 'Qwen-14B', owner: 'Qwen', category: 'LLM', size: '28GB', tags: ['finetune', 'inference'] },
  { name: 'ChatGLM3-6B', owner: 'THUDM', category: 'LLM', size: '12GB', tags: ['finetune', 'inference'] },
  { name: 'Baichuan2-7B', owner: 'baichuan-inc', category: 'LLM', size: '14GB', tags: ['finetune', 'inference'] },
  { name: 'InternLM-20B', owner: 'OpenGVLab', category: 'LLM', size: '40GB', tags: ['finetune', 'inference'] },
  { name: 'Yi-6B', owner: '01-ai', category: 'LLM', size: '12GB', tags: ['finetune', 'inference'] },
];

const MOCK_JOBS = [
  { id: 'job-7f3a2b', model: 'LLaMA2-7B', resources: '2h8g', status: 'Running', created: '2025-03-14 09:00' },
  { id: 'job-9c1d4e', model: 'Qwen-14B', resources: '4h8g', status: 'Completed', created: '2025-03-13 14:30' },
  { id: 'job-2b8f6a', model: 'ChatGLM3-6B', resources: '1h8g', status: 'Completed', created: '2025-03-12 11:20' },
  { id: 'job-5e4a1c', model: 'Baichuan2-7B', resources: '2h8g', status: 'Pending', created: '2025-03-14 10:15' },
];

const MOCK_INFERENCE = [
  { name: 'llama2-chat-svc', model: 'LLaMA2-7B', endpoint: 'https://llama2.example.com/v1', status: 'Ready' },
  { name: 'qwen-api', model: 'Qwen-14B', endpoint: 'https://qwen.example.com/v1', status: 'Ready' },
  { name: 'chatglm-demo', model: 'ChatGLM3-6B', endpoint: 'https://chatglm.example.com/v1', status: 'Ready' },
];

const MOCK_DATASETS = [
  { name: 'alpaca-zh-52k', type: 'Instruction', size: '45MB', records: '52,000' },
  { name: 'sharegpt-en', type: 'Conversation', size: '120MB', records: '90,000' },
  { name: 'dolly-15k', type: 'Instruction', size: '28MB', records: '15,000' },
  { name: 'openorca', type: 'Reasoning', size: '2.1GB', records: '1,200,000' },
];

const MOCK_CLUSTER = [
  { node: 'gpu-node-01', gpuType: 'A100-80GB', gpus: 8, status: 'Ready' },
  { node: 'gpu-node-02', gpuType: 'A100-80GB', gpus: 8, status: 'Ready' },
  { node: 'gpu-node-03', gpuType: 'A100-80GB', gpus: 8, status: 'Ready' },
  { node: 'gpu-node-04', gpuType: 'A100-80GB', gpus: 8, status: 'Ready' },
  { node: 'gpu-node-05', gpuType: 'H100-80GB', gpus: 8, status: 'Ready' },
  { node: 'gpu-node-06', gpuType: 'H100-80GB', gpus: 8, status: 'Ready' },
];

function getStatusClass(status) {
  const map = {
    Running: 'status-running',
    Ready: 'status-running',
    Completed: 'status-completed',
    Pending: 'status-pending',
    Failed: 'status-failed',
  };
  return map[status] || 'status-pending';
}

function renderModels() {
  const tbody = document.getElementById('models-table-body');
  if (!tbody) return;
  tbody.innerHTML = MOCK_MODELS.map(m => `
    <tr>
      <td>${m.name}</td>
      <td>${m.owner}</td>
      <td>${m.category}</td>
      <td>${m.size}</td>
      <td>
        <button class="btn btn-primary btn-sm">Fine-tune</button>
        <button class="btn btn-secondary btn-sm">Deploy</button>
      </td>
    </tr>
  `).join('');
}

function renderJobs() {
  const tbody = document.getElementById('jobs-table-body');
  if (!tbody) return;
  tbody.innerHTML = MOCK_JOBS.map(j => `
    <tr>
      <td>${j.id}</td>
      <td>${j.model}</td>
      <td>${j.resources}</td>
      <td><span class="status-badge ${getStatusClass(j.status)}">${j.status}</span></td>
      <td>${j.created}</td>
    </tr>
  `).join('');
}

function renderInference() {
  const tbody = document.getElementById('inference-table-body');
  if (!tbody) return;
  tbody.innerHTML = MOCK_INFERENCE.map(i => `
    <tr>
      <td>${i.name}</td>
      <td>${i.model}</td>
      <td><code>${i.endpoint}</code></td>
      <td><span class="status-badge ${getStatusClass(i.status)}">${i.status}</span></td>
    </tr>
  `).join('');
}

function renderDatasets() {
  const tbody = document.getElementById('datasets-table-body');
  if (!tbody) return;
  tbody.innerHTML = MOCK_DATASETS.map(d => `
    <tr>
      <td>${d.name}</td>
      <td>${d.type}</td>
      <td>${d.size}</td>
      <td>${d.records}</td>
    </tr>
  `).join('');
}

function renderCluster() {
  const tbody = document.getElementById('cluster-table-body');
  if (!tbody) return;
  tbody.innerHTML = MOCK_CLUSTER.map(c => `
    <tr>
      <td>${c.node}</td>
      <td>${c.gpuType}</td>
      <td>${c.gpus}</td>
      <td><span class="status-badge ${getStatusClass(c.status)}">${c.status}</span></td>
    </tr>
  `).join('');
}

function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.section');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const href = item.getAttribute('href');
      const targetId = href.slice(1);

      navItems.forEach(n => n.classList.remove('active'));
      item.classList.add('active');

      sections.forEach(s => {
        s.classList.remove('active');
        if (s.id === targetId) s.classList.add('active');
      });
    });
  });

  // Hash-based navigation on load
  const hash = window.location.hash.slice(1) || 'overview';
  const targetSection = document.getElementById(hash);
  const targetNav = document.querySelector(`.nav-item[href="#${hash}"]`);

  if (targetSection) {
    sections.forEach(s => s.classList.remove('active'));
    targetSection.classList.add('active');
  }
  if (targetNav) {
    navItems.forEach(n => n.classList.remove('active'));
    targetNav.classList.add('active');
  }
}

function init() {
  renderModels();
  renderJobs();
  renderInference();
  renderDatasets();
  renderCluster();
  setupNavigation();
}

document.addEventListener('DOMContentLoaded', init);
