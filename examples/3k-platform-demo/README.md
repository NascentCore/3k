# 3K Platform Demo

This folder contains the **same web UI as the real 3K platform** (算想云), built in demo mode so you can log in with dummy data and click around without a backend.

- **演示账号（dummy 登录）**: 邮箱 `test@sxwl.ai`，密码 `sxwl666!` — 使用该账号点击「登录」会直接通过，不会请求后端，避免 404。
- **Data**: 登录后的其他 API 可能因无后端而报错或空状态；仅用于界面演示与点击流程。

## 方式一：本地开发带 dummy 登录（推荐）

在项目根目录执行：

```bash
cd ui && npm run start:demo
```

用浏览器打开终端里显示的地址（如 http://localhost:8000），进入 `/user/login`，用上面演示账号登录即可，不会出现 404。

## 方式二：使用本目录静态构建

**必须先**用 demo 模式构建并复制到本目录，否则登录会 404。在项目根目录执行：

```bash
cd ui && npm run deploy:demo
```

然后用静态服务打开本目录，例如：

```bash
npx serve .
# 或: python3 -m http.server 8080
```

打开终端里显示的地址（如 http://localhost:3000），进入登录页，用演示账号 `test@sxwl.ai` / `sxwl666!` 登录。
