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
