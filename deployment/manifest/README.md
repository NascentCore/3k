# 离线安装包制作

## 说明
离线安装包包含 3K 平台所需的所有组件，包括：

- 3K 平台程序包
- 3K 平台依赖包
- 3K 平台配置文件
- 组件依赖的镜像
- 组件依赖的 DEB 包

安装包目录结构如下：

```
├── 3k-artifacts
│   ├── bin             # kubekey 等工具
│   ├── conf            # 3K 平台配置文件
│   ├── cli             # 3kctl 命令行工具
│   ├── deploy          # 部署组件所需的 yaml 配置文件
│   ├── helm-charts     # 组件 charts 包
│   ├── helm-plugins    # helm 插件
│   └── packages        # 组件依赖的安装包（deb、images）
```

3K 平台程序包、配置文件、依赖包、镜像、DEB 包等组件均放在同一目录下，方便统一管理。

## 制作离线安装包
1. 克隆 3K 平台仓库到本地：
```
git clone https://github.com/NascentCore/3k.git
```

2. 进入 3K 平台仓库的 `deployment/manifest` 目录，执行 `make` 命令，生成离线安装包：
```
cd 3k/deployment/manifest
make
```

3. 制作完成后，会在当前目录生成 `3k-artifacts.tar.gz` 文件，即为离线安装包。

