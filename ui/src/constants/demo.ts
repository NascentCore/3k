/**
 * Demo 模式下的演示账号与用户信息
 */
export const DEMO_CREDENTIALS = {
  email: 'test@sxwl.ai',
  password: 'sxwl666!',
} as const;

export const DEMO_USER: API.CurrentUser = {
  name: 'Demo User',
  username: 'Demo User',
  user_id: 'demo',
  avatar: undefined,
};

/** Demo 模式下 LLM 仓库的公共/私有模型列表（基于 Hugging Face 文本生成热门模型） */
export const DEMO_MODELS = {
  public_list: [
    { id: 'demo-deepseek-v3.2', name: 'deepseek-ai/DeepSeek-V3.2', owner: 'deepseek-ai', category: 'Text Generation', size: 1.2e12, tag: ['finetune', 'inference'] },
    { id: 'demo-minimax-m2.5', name: 'MiniMaxAI/MiniMax-M2.5', owner: 'MiniMaxAI', category: 'Text Generation', size: 4.5e11, tag: ['finetune', 'inference'] },
    { id: 'demo-glm5', name: 'zai-org/GLM-5', owner: 'zai-org', category: 'Text Generation', size: 1.4e12, tag: ['inference'] },
    { id: 'demo-qwen3-coder', name: 'Qwen/Qwen3-Coder-Next', owner: 'Qwen', category: 'Text Generation', size: 1.6e11, tag: ['finetune', 'inference'] },
    { id: 'demo-llama-3.1', name: 'meta-llama/Llama-3.1-8B-Instruct', owner: 'meta-llama', category: 'Text Generation', size: 1.6e10, tag: ['finetune', 'inference'] },
    { id: 'demo-step-3.5', name: 'stepfun-ai/Step-3.5-Flash', owner: 'stepfun-ai', category: 'Text Generation', size: 4e11, tag: ['inference'] },
    { id: 'demo-sarvam-105b', name: 'sarvamai/sarvam-105b', owner: 'sarvamai', category: 'Text Generation', size: 2.1e11, tag: ['finetune', 'inference'] },
    { id: 'demo-nemotron-120b', name: 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16', owner: 'nvidia', category: 'Text Generation', size: 2.4e11, tag: ['inference'] },
    { id: 'demo-nanbeige-3b', name: 'Nanbeige/Nanbeige4.1-3B', owner: 'Nanbeige', category: 'Text Generation', size: 6e9, tag: ['finetune', 'inference'] },
    { id: 'demo-gpt-oss-120b', name: 'openai/gpt-oss-120b', owner: 'openai', category: 'Text Generation', size: 2.4e11, tag: ['inference'] },
  ],
  user_list: [
    { id: 'demo-user-deepseek-mini', name: 'demo/deepseek-ai/DeepSeek-V3.2-mini', owner: 'Demo User', category: 'Text Generation', size: 8e10, tag: ['finetune', 'inference'] },
    { id: 'demo-user-llama-ft', name: 'demo/meta-llama/Llama-3.1-8B-custom', owner: 'Demo User', category: 'Text Generation', size: 1.6e10, tag: ['finetune', 'inference'] },
  ],
};

/** Demo 模式下 LLM 适配器列表（公共 / 私有） */
export const DEMO_ADAPTERS = {
  public_list: [
    { id: 'demo-adapter-lora-7b', name: 'lora/llama-3.1-7b-sft', size: 256 * 1024 * 1024 },
    { id: 'demo-adapter-qlora-8b', name: 'qlora/qwen2.5-8b-instruct', size: 128 * 1024 * 1024 },
    { id: 'demo-adapter-lora-deepseek', name: 'lora/deepseek-v3-lite', size: 512 * 1024 * 1024 },
    { id: 'demo-adapter-ia3', name: 'ia3/glm-4-9b-chat', size: 64 * 1024 * 1024 },
    { id: 'demo-adapter-adapters-hub', name: 'AdapterHub/bert-base-uncased-pf-sst2', size: 1.2 * 1024 * 1024 },
  ],
  user_list: [
    { id: 'demo-user-adapter-custom', name: 'demo/lora/my-custom-7b', size: 192 * 1024 * 1024 },
    { id: 'demo-user-adapter-finetuned', name: 'demo/qlora/step-3.5-finetuned', size: 96 * 1024 * 1024 },
  ],
};

/** Demo 模式下 LLM 应用列表（New API 支持的 AI 应用，见 https://www.newapi.ai/en/docs/apps） */
export const DEMO_APPS = [
  { app_id: 'demo-aionui', app_name: 'AionUi', desc: 'Free, local, open-source 24/7 Cowork and OpenClaw for Gemini CLI, Claude Code, Codex, OpenCode, Qwen Code, Goose CLI, Auggie, and more.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/aionui' },
  { app_id: 'demo-cc-switch', app_name: 'CC Switch', desc: 'One-click fill from New API token page; configure Claude, Codex, or Gemini with main and variant models in the popup.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/cc-switch' },
  { app_id: 'demo-cherry-studio', app_name: 'Cherry Studio', desc: 'A powerful AI assistant client that supports multi-model conversations.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/cherry-studio' },
  { app_id: 'demo-openclaw', app_name: 'OpenClaw', desc: 'A self-hosted AI assistant platform — install OpenClaw, connect to New API, and manage agents across Feishu, Discord, Slack, and more.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/openclaw' },
  { app_id: 'demo-fluent-read', app_name: 'Fluent Read', desc: 'An AI-powered smart reading and translation assistant.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/fluent-read' },
  { app_id: 'demo-langbot', app_name: 'LangBot', desc: 'A large language model-based chatbot framework.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/langbot' },
  { app_id: 'demo-luna-translator', app_name: 'Luna Translator', desc: 'A real-time translation tool for games and documents.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/luna-translator' },
  { app_id: 'demo-astrbot', app_name: 'AstrBot', desc: 'An open-source, all-in-one Agent chatbot platform.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/astrbot' },
  { app_id: 'demo-claude-code', app_name: 'Claude Code', desc: 'An Anthropic Claude-powered code editor integration.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/claude-code' },
  { app_id: 'demo-codex-cli', app_name: 'Codex CLI', desc: 'A command-line interface AI code assistant tool.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/codex-cli' },
  { app_id: 'demo-factory-droid-cli', app_name: 'Factory Droid CLI', desc: 'An AI agent tool for automating workflows.', isReference: true, docUrl: 'https://www.newapi.ai/en/docs/apps/factory-droid-cli' },
];

/** Demo 模式下数据集列表（公共 / 私有） */
const demoDatasetMeta = (total: number, evalSet: boolean, preview: object[]) => ({
  total,
  eval: evalSet,
  preview: JSON.stringify(preview),
});

export const DEMO_DATASETS = {
  public_list: [
    {
      id: 'demo-dataset-alpaca',
      name: 'alpaca-gpt4-52k',
      size: 128 * 1024 * 1024,
      is_public: true,
      meta: demoDatasetMeta(52000, false, [
        { instruction: 'Summarize the following.', input: 'Long text...', output: 'Summary.' },
        { instruction: 'Translate to French.', input: 'Hello world', output: 'Bonjour le monde' },
      ]),
    },
    {
      id: 'demo-dataset-mmlu',
      name: 'MMLU-pro-val',
      size: 256 * 1024 * 1024,
      is_public: true,
      meta: demoDatasetMeta(12000, true, [
        { question: 'What is 2+2?', choices: ['3', '4', '5'], answer: '4' },
        { question: 'Capital of France?', choices: ['London', 'Paris', 'Berlin'], answer: 'Paris' },
      ]),
    },
    {
      id: 'demo-dataset-sft-qa',
      name: 'sft-qa-10k',
      size: 64 * 1024 * 1024,
      is_public: true,
      meta: demoDatasetMeta(10000, false, [
        { q: 'What is machine learning?', a: 'A subset of AI that learns from data.' },
      ]),
    },
  ],
  user_list: [
    {
      id: 'demo-user-dataset-custom',
      name: 'demo/custom-qa-5k',
      size: 48 * 1024 * 1024,
      is_public: false,
      meta: demoDatasetMeta(5000, false, [
        { prompt: 'Explain X', response: 'X is...' },
      ]),
    },
    {
      id: 'demo-user-dataset-eval',
      name: 'demo/my-eval-set',
      size: 32 * 1024 * 1024,
      is_public: false,
      meta: demoDatasetMeta(2000, true, [
        { question: 'Q1', options: ['A', 'B'], correct: 'A' },
      ]),
    },
  ],
};

/** Demo 模式下开发实验室镜像列表（GET /api/job/jupyter/image 的 data） */
export const DEMO_JOB_JUPYTER_IMAGES = {
  data: [
    {
      image_name: 'demo-jupyter-cuda12',
      created_at: new Date(Date.now() - 86400 * 7 * 1000).toISOString(),
      updated_at: new Date(Date.now() - 86400 * 2 * 1000).toISOString(),
    },
    {
      image_name: 'demo-jupyter-pytorch',
      created_at: new Date(Date.now() - 86400 * 14 * 1000).toISOString(),
      updated_at: new Date(Date.now() - 86400 * 5 * 1000).toISOString(),
    },
  ],
};

/** Demo 模式下开发实验室（JupyterLab）实例列表 */
export const DEMO_JUPYTERLAB_INSTANCES = [
  {
    instance_name: 'demo-lab-01',
    cpu_count: 8,
    memory: 32 * 1024 * 1024 * 1024, // 32 GiB in bytes
    gpu_product: 'NVIDIA A100 40GB',
    status: 'running',
    job_name: 'demo-jupyterlab',
    url: 'https://jupyterlab.example.com/demo',
  },
];

/** Demo 模式下任务详情：推理服务列表（GET /api/job/inference 的 data） */
export const DEMO_JOB_DETAIL_INFERENCE = [
  {
    service_name: 'demo-inference-chat-01',
    model_name: 'deepseek-ai/DeepSeek-V3.2',
    gpu_model: 'NVIDIA A100 40GB',
    gpu_count: 1,
    start_time: new Date(Date.now() - 86400 * 1000).toISOString(),
    end_time: undefined,
    create_time: new Date(Date.now() - 86400 * 1000).toISOString(),
    status: 'running',
    model_category: 'chat',
    url: '/chat-trial?model=deepseek-ai/DeepSeek-V3.2',
    api: `${typeof window !== 'undefined' ? window.location?.origin : ''}/api/v1/chat/completions`,
  },
  {
    service_name: 'demo-inference-embed-01',
    model_name: 'BAAI/bge-m3',
    gpu_model: 'NVIDIA L40S',
    gpu_count: 1,
    start_time: new Date(Date.now() - 3600 * 1000).toISOString(),
    end_time: undefined,
    create_time: new Date(Date.now() - 3600 * 1000).toISOString(),
    status: 'running',
    model_category: 'embedding',
    url: `${typeof window !== 'undefined' ? window.location?.origin : ''}/api/v1/embeddings`,
    api: undefined,
  },
  {
    service_name: 'demo-inference-finished',
    model_name: 'Qwen/Qwen3-Coder-Next',
    gpu_model: 'NVIDIA A100 40GB',
    gpu_count: 2,
    start_time: new Date(Date.now() - 86400 * 2 * 1000).toISOString(),
    end_time: new Date(Date.now() - 86400 * 1000).toISOString(),
    create_time: new Date(Date.now() - 86400 * 2 * 1000).toISOString(),
    status: 'stopped',
    model_category: 'chat',
    url: '',
    api: undefined,
  },
];

/** Demo 模式下任务详情：训练任务列表（GET /api/job/training 的 content） */
export const DEMO_JOB_DETAIL_USER_JOB = [
  {
    jobName: 'demo-finetune-01',
    pretrainedModelName: 'meta-llama/Llama-3.1-8B-Instruct',
    jobType: 'Finetune',
    gpuType: 'NVIDIA A100 40GB',
    gpuNumber: 2,
    createTime: new Date(Date.now() - 3600 * 3 * 1000).toISOString(),
    updateTime: undefined,
    workStatus: 1,
    status: 'running',
    tensor_url: '#',
    userId: 'demo',
  },
  {
    jobName: 'demo-gpujob-01',
    pretrainedModelName: 'stepfun-ai/Step-3.5-Flash',
    jobType: 'GPUJob',
    gpuType: 'NVIDIA L40S',
    gpuNumber: 1,
    createTime: new Date(Date.now() - 86400 * 2 * 1000).toISOString(),
    updateTime: new Date(Date.now() - 86400 * 1000).toISOString(),
    workStatus: 8,
    status: 'succeeded',
    tensor_url: '#',
    userId: 'demo',
  },
];

/** Demo 模式下推理试用页面对用户消息的随机中文回复列表 */
export const DEMO_CHAT_REPLIES: string[] = [
  '好的，我理解您的问题。根据目前的信息，建议您先确认一下具体的使用场景，这样我可以给出更精准的建议。',
  '感谢您的提问。这个问题可以从几个方面来看：一是流程上的优化，二是权限与配置的核对。您方便说一下当前遇到的具体现象吗？',
  '您提到的这一点很重要。在实际使用中，建议先检查网络和账号权限，多数类似情况都与这两项有关。',
  '我这边根据您的描述，建议您先尝试刷新页面或重新登录；若问题依旧，可以把报错信息或截图发过来，我帮您一起排查。',
  '明白。这类需求一般可以通过配置或脚本实现，您可以说一下希望达到的效果（例如自动重试、告警等），我按您的目标给出一套可行方案。',
  '收到。从您描述来看，更可能是配置或环境差异导致的，建议对照文档再核对一遍相关参数和依赖版本。',
  '好的，已记录。后续如果您有新的报错信息或日志，可以继续发给我，我们一起定位。',
  '理解。建议您先确认当前账号是否有对应权限，以及所用环境（开发/测试/生产）是否一致。',
  '这个问题比较常见。一般先看服务是否正常、接口地址和参数是否正确，再排查客户端缓存或版本。',
  '可以。您把具体错误提示或操作步骤发一下，我按步骤帮您分析可能的原因和解决办法。',
];

/** Demo 模式下推理试用（Playground）模型列表（GET /api/job/inference/playground 的 data） */
export const DEMO_PLAYGROUND_MODELS = [
  {
    model_name: 'deepseek-ai/DeepSeek-V3.2',
    url: '/chat-trial?model=deepseek-ai/DeepSeek-V3.2',
    base_url: `${typeof window !== 'undefined' ? window.location?.origin : ''}/api/v1`,
  },
  {
    model_name: 'Qwen/Qwen3-Coder-Next',
    url: '/chat-trial?model=Qwen/Qwen3-Coder-Next',
    base_url: `${typeof window !== 'undefined' ? window.location?.origin : ''}/api/v1`,
  },
  {
    model_name: 'stepfun-ai/Step-3.5-Flash',
    url: '/chat-trial?model=stepfun-ai/Step-3.5-Flash',
    base_url: `${typeof window !== 'undefined' ? window.location?.origin : ''}/api/v1`,
  },
];

/** Demo 模式下 GPU 类型列表（GET /api/resource/gpus 的返回值，用于集群/任务表单） */
export const DEMO_GPUS = [
  { gpuProd: 'NVIDIA A100 40GB', gpuAllocatable: 8 },
  { gpuProd: 'NVIDIA L40S', gpuAllocatable: 8 },
];

/** Demo 模式下集群 CPods 列表（GET /api/cluster/cpods 返回的 data 按 cpod_id 分组后的结构） */
const _demoCpodNodes = (cpodId: string, cpodName: string, nodeNames: string[], gpuProd: string, gpuMemGb: number) =>
  nodeNames.map((node_name, i) => ({
    cpod_id: cpodId,
    cpod_name: cpodName,
    node_name,
    node_type: 'control-plane,worker',
    gpu_prod: gpuProd,
    gpu_mem: gpuMemGb * 1024 * 1024 * 1024,
    gpu_total: 8,
    gpu_allocatable: i === 0 ? 6 : 8,
  }));

export const DEMO_CLUSTER_CPODS: Record<string, Array<{ cpod_id: string; cpod_name: string; node_name: string; node_type: string; gpu_prod: string; gpu_mem: number; gpu_total: number; gpu_allocatable: number }>> = {
  'demo-cpod-01': _demoCpodNodes('demo-cpod-01', '演示集群 A', ['demo-node-01', 'demo-node-02', 'demo-node-03'], 'NVIDIA A100 40GB', 40),
  'demo-cpod-02': _demoCpodNodes('demo-cpod-02', '演示集群 B', ['demo-node-04', 'demo-node-05'], 'NVIDIA L40S', 48),
};
