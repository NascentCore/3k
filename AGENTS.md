# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is the 3K Platform (三千平台) by NascentCore — a cloud-native LLM training/inference platform. It is a monorepo with two Go modules, a React UI, and several other components.

### Key Services for Development

| Service | Directory | Run Command |
|---|---|---|
| Root Go module (scheduler, gateway, sxwlctl) | `/workspace` | `go build ./...` |
| CPod Operator Go module | `/workspace/cpodoperator` | `go build ./...` |
| Web UI (React/Ant Design Pro) | `/workspace/ui` | `yarn start` (dev on port 8000) |

### Lint / Test / Build

- **Go lint**: `golangci-lint run --config=.github/linters/.golangci.yml --timeout=10m` (from repo root). Also `tools/lint.sh` runs golangci-lint + additional checks.
- **Go build (root)**: `go build -mod=readonly -v ./...`
- **Go build (operator)**: `cd cpodoperator && go build -v ./...`
- **Go test (root)**: `go test -v $(go list ./... | grep -v "/e2e")`
- **Go test (operator)**: `cd cpodoperator && go test $(go list ./... | grep -v "litellm")` — the `litellm` package tests require an external service and will fail without it.
- **UI lint**: `cd ui && npx eslint --cache --ext .js,.jsx,.ts,.tsx ./src`
- **UI TypeScript check**: `cd ui && npx tsc --noEmit` (pre-existing errors in the codebase)
- **UI dev server**: `cd ui && yarn start:dev` (runs on port 8000 with MOCK=none)

### Gotchas

- The `cpodoperator/internal/controller` tests have a pre-existing compilation error (`InferenceReconciler` missing `DeployWebUI`/`DeployWebUIIngress` methods). Exclude that package from tests or expect failures.
- The `cpodoperator/pkg/provider/litellm` tests require network access to `playground.llm.sxwl.ai:30005`. Exclude from local test runs.
- TypeScript strict checks (`tsc --noEmit`) report ~22 pre-existing errors in the UI code.
- ESLint reports ~97 pre-existing warnings/errors in the UI code.
- The scheduler and gateway services require MySQL, Alibaba Cloud OSS credentials, and other env vars to run. They are not runnable locally without those external dependencies.
- The `cpodoperator` module uses Go 1.22 while the root module uses Go 1.21. The system Go 1.22 handles both.
- `golangci-lint` v1.54 is installed at `~/go/bin/golangci-lint`. Ensure `~/go/bin` is on PATH.
