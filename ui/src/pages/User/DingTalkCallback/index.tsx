// src/pages/User/DingTalkCallback/index.tsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import { saveToken } from '@/utils';
import { apiDingtalkUserInfo } from '@/services'; // 假设你有一个服务方法去调用 /api/dingtalk/userinfo

const DingTalkLoginCallback: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const code = searchParams.get('code');

    if (!code) {
      message.error('扫码登录失败，缺少code');
      setLoading(false);
      return;
    }

    // 发起请求，获取用户信息并保存 JWT
    const fetchUserInfo = async () => {
      try {
        const response = await apiDingtalkUserInfo(code);
        if (response?.token) {
          // 保存 token 到 cookie
          saveToken(response.token);
          message.success('登录成功');
          // 跳转到主页或者之前用户试图访问的页面
          const urlParams = new URL(window.location.href).searchParams;
          navigate(urlParams.get('redirect') || '/');
        } else {
          message.error('获取用户信息失败');
        }
      } catch (error) {
        message.error('登录失败，请稍后重试');
        console.error('Error fetching user info:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, [navigate]);

  return (
    <div style={{ textAlign: 'center', marginTop: 50 }}>
      {loading ? <p>正在处理登录，请稍候...</p> : <p>登录失败，请重试。</p>}
    </div>
  );
};

export default DingTalkLoginCallback;
