import { PageContainer } from '@ant-design/pro-components';
import { Select, Button, message } from 'antd';
import React, { useEffect, useState } from 'react';
import { apiInferencePlayground } from '@/services';
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/prism';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import styles from './index.less';
import { CopyOutlined } from '@ant-design/icons';
import { useIntl } from '@umijs/max';

interface PlaygroundModel {
  model_name: string;
  url: string;
  base_url: string;
}

const Playground: React.FC = () => {
  const intl = useIntl();
  const [models, setModels] = useState<PlaygroundModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>();
  const [chatUrl, setChatUrl] = useState<string>();
  const [showCopyButton, setShowCopyButton] = useState(false);

  const fetchModels = async () => {
    try {
      const response = await apiInferencePlayground();
      setModels(response.data || []);
      if (response.data?.length > 0) {
        setSelectedModel(response.data[0].model_name);
        // setChatUrl(response.data[0].url);
        setChatUrl(`/chat-trial?model=${response.data[0].model_name}`);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleModelChange = (value: string) => {
    setSelectedModel(value);
    const selectedModelData = models.find((m) => m.model_name === value);
    setChatUrl(`/chat-trial?model=${selectedModelData?.model_name}`);
  };

  const sampleCode = `from openai import OpenAI

client = OpenAI(
    base_url="${window.location.origin}/api/v1",
    api_key="dummy", # 算想云token
)

response = client.chat.completions.create(
    model="${selectedModel}",
    messages=[
        {"role": "user", "content": "How to learn python?"}
    ],
    max_tokens=200,
    temperature=0.7,
    top_p=1,
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
`;

  const handleCopy = () => {
    navigator.clipboard
      .writeText(sampleCode)
      .then(() => {
        message.success(intl.formatMessage({ id: 'playground.copy.success' }));
      })
      .catch(() => {
        message.error(intl.formatMessage({ id: 'playground.copy.failed' }));
      });
  };

  return (
    <PageContainer>
      <div className={styles.container}>
        <div className={styles.topSection}>
          <Select
            style={{ width: 200 }}
            value={selectedModel}
            onChange={handleModelChange}
            options={models.map((model) => ({
              label: model.model_name,
              value: model.model_name,
            }))}
          />
        </div>
        <div className={styles.mainContent}>
          <div className={styles.leftSection}>
            <div className={styles.chatContainer}>
              {chatUrl && (
                <iframe
                  src={chatUrl}
                  style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    transform: 'scale(1)',
                    transformOrigin: '0 0',
                    minWidth: '100%',
                    minHeight: '100%',
                  }}
                  sandbox="allow-same-origin allow-scripts allow-popups allow-forms allow-modals"
                  allow="camera *; microphone *"
                  referrerPolicy="origin"
                  onLoad={() => console.log('iframe loaded')}
                  onError={(e) => console.error('iframe error:', e)}
                />
              )}
            </div>
          </div>
          <div
            className={styles.rightSection}
            onMouseEnter={() => setShowCopyButton(true)}
            onMouseLeave={() => setShowCopyButton(false)}
          >
            {showCopyButton && (
              <Button
                className={styles.copyButton}
                icon={<CopyOutlined />}
                size="small"
                onClick={handleCopy}
              >
                {intl.formatMessage({ id: 'playground.copy.button' })}
              </Button>
            )}
            <SyntaxHighlighter
              language="python"
              style={vscDarkPlus}
              customStyle={{
                margin: 0,
                borderRadius: '8px',
              }}
              codeTagProps={{
                style: {
                  lineHeight: '1',
                },
              }}
            >
              {sampleCode}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>
    </PageContainer>
  );
};

export default Playground;
