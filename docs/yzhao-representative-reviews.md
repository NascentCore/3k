# Representative pull requests

The five items that best show yzhao’s seniority are the Job abstraction on #239, the GPU-metrics scope on #128, the testable ModelStorage split on #591, the single PVC-name function on #236, and the `podTmplConfig` struct on #334.

## Rank

1. **[#239](https://github.com/NascentCore/3k/pull/239#discussion_r1378679450)** — abstract `Job` before Kubeflow Training Operator adds more types ([issue #195](https://github.com/NascentCore/3k/issues/195))
   - Names the failure mode: MPIJob, PyTorchJob, and inference will each grow their own copy.
   - Asks for the abstraction from the MPIJob implementation plus the distributed-training requirements, before more job kinds land.
   - Companion note on the same pull request: shrink the function so any engineer in the company can change it.
2. **[#128](https://github.com/NascentCore/3k/pull/128#discussion_r1353786803)** — MVP GPU report is three counts: total, used, free
   - Stops the open-ended `GPUState` schema debate at the scheduling contract the market manager needs.
   - Assigns the full device-telemetry design to Glen and the follow-up implementation to congpeiqing.
3. **[#591](https://github.com/NascentCore/3k/pull/591#discussion_r1497374062)** — `generateModelStorageCR` stays pure; `createModelstorage` performs `Client.Create`
   - Separates data construction from the API call so the builder can be unit-tested.
   - The split is still in `cpodjob_controller.go`: `generateModelstorage` builds the object, the caller does `Client.Create`.
4. **[#236](https://github.com/NascentCore/3k/pull/236#discussion_r1377591864)** — `GetJobModelOutputPVCName(jobID)` is the only definition of the model-output PVC name
   - Create and delete call the same function, so a rename cannot drift.
   - Cites Go inlining, so the helper has no call cost.
   - Same review: `StorageClass` and `ReadWriteMany` go in `pkg/utils/consts/k8s.go`.
5. **[#334](https://github.com/NascentCore/3k/pull/334#discussion_r1411893298)** — the `podTemplate` argument list becomes `podTmplConfig`
   - Fields: dataset PVC and path, pretrained-model PVC and path, checkpoint path, model-save path, `withSHM`.
   - A new volume or path is one field, not a new parameter on every caller.

## Why this order

Rank 1 and 2 set direction for other people. Rank 3 is that same habit applied to the training reconciler, and the code still follows it. Rank 4 and 5 are the kept API-shape reviews: one name for a string that is used in more than one place, one struct when a parameter list becomes a configuration object.

## Supporting authored pull requests

The earlier authored set shows the same habits at smaller scale. Those support the record. The five above are the ones a staff reader would use to judge seniority.

- **[#133](https://github.com/NascentCore/3k/pull/133)** — drop `market-manager/go.mod`; that tree joins the root module.
- **[#114](https://github.com/NascentCore/3k/pull/114)** — golangci-lint workflow and the first cleanup of `cpodmanager`, `job.go`, and the MPIJob client.
- **[#216](https://github.com/NascentCore/3k/pull/216)** — `K8S_LABEL_NV_GPU_PRODUCT` and `K8S_LABEL_NV_GPU_PRESENT` in `pkg/utils/consts/k8s.go`.
- **[#473](https://github.com/NascentCore/3k/pull/473)** — `pkg/utils/config` file and env readers, each with a test.
- **[#180](https://github.com/NascentCore/3k/pull/180)** — CephFS volume on the BERT MPIJob, applied in namespace `bert` on the LY cluster.
