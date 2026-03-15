# 3K Platform Demo

本目录为 3K 平台（算想云）的 demo 构建，使用演示账号可无后端登录浏览界面。

**演示账号**：`test@sxwl.ai` / `sxwl666!`

## 最简步骤

```bash
cd ui && npm run start:demo
```

浏览器打开终端里的地址（如 <http://localhost:8000>），到登录页用上述账号登录。

打开终端里的地址，用同一账号登录。

# 安装 Node.js 最简步骤

如需用「方式一」本地服务器打开演示，或运行其他前端/脚本工具，需要先安装 Node。

**国内用户推荐**：使用国内镜像下载更快  
→ [阿里云 Node 镜像](https://developer.aliyun.com/mirror/nodejs-release) — 选 **LTS** 版本，再选对应系统（Windows 选 .msi，macOS 选 .pkg）下载。

---

## macOS

1. 打开 [https://nodejs.org](https://nodejs.org)（国内可用 [阿里云 Node 镜像](https://developer.aliyun.com/mirror/nodejs-release)），下载 **LTS** 的 `.pkg` 安装包。
2. 下载完成后双击 `.pkg` 安装包，按提示一路「继续」完成安装。
3. 打开「终端」（在「应用程序 → 实用工具」里），输入：
   ```bash
   node -v
   ```
   若显示版本号（如 `v20.x.x`），说明安装成功。

---

## Windows

1. 打开 [https://nodejs.org](https://nodejs.org)（国内可用 [阿里云 Node 镜像](https://developer.aliyun.com/mirror/nodejs-release)），下载 **LTS** 的 `.msi` 安装包。
2. 双击下载的 `.msi` 安装包，按提示下一步完成安装（可勾选「自动安装必要工具」）。
3. 关闭并重新打开「命令提示符」或 PowerShell，输入：
   ```bash
   node -v
   ```
   若显示版本号，说明安装成功。

---

安装好后，在演示文件夹里打开终端/命令提示符，执行 `npx serve -p 8000 .`，浏览器访问 <http://localhost:8000> 即可。若只需双击 **index.html** 演示，则**不需要**安装 Node。
