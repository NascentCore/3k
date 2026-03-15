# Agent memory

## Learned User Preferences

- Prefer 3k platform demo UI in Chinese (e.g. default zh-CN in demo mode).

## Learned Workspace Facts

- 3k-platform-demo: build with base `/` and publicPath `/`; serve only the demo directory (e.g. `npx serve examples/3k-platform-demo`) and open the root URL to avoid asset 404s.
- Demo mode: set REACT_APP_DEMO and UMI_APP_DEMO at build; inject via config define; intercept apiAuthInfo and apiAuthLogin in services when in demo so no backend requests are made.
- Run interactive demo with `cd ui && npm run start:demo`; demo credentials are in ui/src/constants/demo.ts and examples/3k-platform-demo/README.md.
- Sidebar logo: use `/favicon.ico` (same-origin); place favicon in ui/public so it is included in build and works when served from localhost.
