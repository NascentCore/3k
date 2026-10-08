---
# B. Authored pull requests — NascentCore/3k

This section is the convergence sample. The system is an LLM compute / GPU cluster platform (cpod manager, Kubeflow MPIJob training, Ceph storage, CI). The transferable part is senior ownership: raising engineering standards, making distributed training paths real on cluster storage, automating node bring-up, and collapsing module / config surface area so the platform stays operable.

`yzhao-2023` (Yaxiong Zhao) has 79 merged pull requests on 3k. The five below are the ones kept for this packet.

## Recommendation

These PRs show an engineer who:

- Treats **engineering standards** (lint, test presence, check aggregation) as product infrastructure, not optional polish
- Owns **distributed ML + cluster storage** end to end: Kubeflow MPIJob, DeepSpeed, CephFS mounts, durable checkpoints
- Prefers **ops automation** that makes GPU/VM nodes reproducible instead of tribal knowledge
- Writes **shared platform primitives** (config, modules) with tests and explicit intent for reuse
- Makes **architecture hygiene** decisions: consolidate Go modules, delete unused paths, keep the dependency graph honest

## How the five were chosen

Of 79 merged pull requests authored by `yzhao-2023`, these five were kept because the PR showed standards leadership, K8s/storage/training depth, infra automation, shared-library design, or monorepo module judgment, and because the PR scope and follow-through made that visible. Rename-only, docs-only, menu reorder, and opportunistic large-delete cleanups were set aside.

## Index

| # | Pull request | Scale | Merged | One-line change |
| --- | --- | --- | --- | --- |
| 1 | [#114](https://github.com/NascentCore/3k/pull/114) | +132 / −46 · 11 files | 2023-10-03 | Introduce golangci-lint as CI quality gate |
| 2 | [#185](https://github.com/NascentCore/3k/pull/185) | +72 / −30 · 3 files | 2023-10-26 | Persist Bert MPIJob checkpoints on CephFS |
| 3 | [#72](https://github.com/NascentCore/3k/pull/72) | +122 / −0 · 2 files | 2023-09-11 | Automate Linux VM data-disk initialization |
| 4 | [#473](https://github.com/NascentCore/3k/pull/473) | +84 / −7 · 6 files | 2024-01-13 | Unified config APIs for file and env vars |
| 5 | [#133](https://github.com/NascentCore/3k/pull/133) | +188 / −258 · 10 files | 2023-10-12 | Fold market-manager into root Go module |

Read as platform ownership samples: **#114** (standards), **#185** (training path), **#72** (node ops). #473 and #133 are the abstraction / architecture samples beside that spine.

## 1. Raise Go quality gates with brownfield-aware lint CI

**PR:** [#114](https://github.com/NascentCore/3k/pull/114)

**Problem.** The Go codebase lacked a focused, enforceable lint workflow. Broad super-linter style tooling either missed Go-specific issues or created noise that the team could not act on. New regressions could land without a mechanical gate.

**Change.** Added a `golangci-lint` GitHub Actions workflow and `.golangci.yml` configuration. Enabled `new: true` so only new issues fail the build on an existing large tree. Fixed lint findings across `manager/` packages (auth, cluster client, job, communication). Documented tradeoffs (e.g. disabling `depguard` during rapid prototyping).

**Signal.**

- Names a quality gap and installs a gate the team can live with
- Brownfield pragmatism: do not demand a full historical clean before protecting HEAD
- Touches CI config and production Go packages as one change

## 2. Make distributed training durable on cluster storage

**PR:** [#185](https://github.com/NascentCore/3k/pull/185) · builds on [#180](https://github.com/NascentCore/3k/pull/180)

**Problem.** Bert / DeepSpeed MPIJob examples could run on GPUs but checkpoint durability was unclear. Without CephFS-backed paths, training progress stayed ephemeral to the pod filesystem and was hard to resume or inspect.

**Change.** Extended the Bert MPIJob example so training writes checkpoints to a mounted CephFS path. Updated Dockerfile, `mpi_bert_ds.yaml`, and `train_bert_ds.py` together. Continues the CephFS mount work proven on the LY cluster in #180.

**Signal.**

- Crosses Kubeflow MPIJob, DeepSpeed, and Ceph storage as one path
- Treats checkpoint durability as part of the training contract, not an afterthought
- Review engagement on design choices (e.g. what belongs in the image vs mounted storage)

## 3. Automate GPU/VM data-disk bring-up

**PR:** [#72](https://github.com/NascentCore/3k/pull/72)

**Problem.** Cluster nodes with attached data disks required manual partition, format, mount, and fstab work. That process is error-prone and does not scale when GPU VMs are provisioned repeatedly.

**Change.** Added `devops/LinuxVMDataDiskAutoInitialize.sh` (root-gated interactive script) plus devops README pointers. Covers disk selection, unmount/format safety prompts, partitioning, filesystem creation, mount, and fstab persistence.

**Signal.**

- Owns the ops layer that makes nodes reproducible
- Encodes safety checks (root privilege, LVM refusal, OS-disk exclusion, remount confirmation)
- Infrastructure automation as first-class platform work

## 4. Unified config reading for files and environment

**PR:** [#473](https://github.com/NascentCore/3k/pull/473)

**Problem.** Future services needed a common way to load configuration from files and environment variables. Ad-hoc readers would diverge and make env/file overrides inconsistent.

**Change.** Implemented `pkg/utils/config` readers for file (`key: value` parsing) and env vars, with unit tests. Framed explicitly as shared APIs for later code to consume.

**Signal.**

- Platform primitive before feature sprawl
- Tests travel with the API surface
- Small blast radius, clear reuse intent

## 5. Collapse market-manager into the root Go module

**PR:** [#133](https://github.com/NascentCore/3k/pull/133)

**Problem.** `market-manager` carried a separate `go.mod` / `go.sum` while living in the same monorepo as the rest of the platform. Split modules raise version skew, duplicate dependency work, and confuse build ownership.

**Change.** Removed the nested module and aligned the root `go.mod` with the real dependency set (Gin, GORM, K8s client libraries, swagger). Net deletion of dependency and module surface while keeping market-manager code on the shared module graph.

**Signal.**

- Monorepo judgment: one module unless boundaries earn their keep
- Dependency graph treated as architecture, not accident
- Entropy reduction without a product-facing rewrite

## Competency map

| Competency | Pull requests |
| --- | --- |
| Engineering standards and CI ownership | #114 (supported by #119, #93, #125) |
| Distributed training and cluster storage | #185 (supported by #180) |
| Production infra / node ops | #72 |
| Shared library and API design | #473 (supported by #481) |
| Module and dependency architecture | #133 |
| Safe, test-backed foundational work | #473, #481 |

## Secondary samples

- [#180](https://github.com/NascentCore/3k/pull/180) — complete CephFS mount into MPIJob (storage path that #185 builds on)
- [#119](https://github.com/NascentCore/3k/pull/119) — remove super-linter in favor of focused golangci-lint
- [#93](https://github.com/NascentCore/3k/pull/93) — CI check that every substantial `.go` file has a `_test.go` (with escape hatches)
- [#481](https://github.com/NascentCore/3k/pull/481) — testing FS helpers for temp files and content fixtures
- [#46](https://github.com/NascentCore/3k/pull/46) — Dockerfile cleanup and base/dev image rebuild for Bert DeepSpeed

## Bottom line

The strongest signal in this section is senior platform ownership on an early LLM compute stack: install quality gates the team can adopt, prove distributed training against real cluster storage, automate node disk bring-up, and keep shared config and Go modules coherent so the system stays operable as it grows.
