package kubeflowmpijob

import (
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
)

// TODO: 根据实际情况增加或者删除一些字段
// 表示创建一个KubeFLowMPIJob需要知道的信息.
type MPIJob struct {
	Name                 string // will be used as metadata.name
	Namespace            string // k8s namespace
	Image                string // docker image
	DataPath             string // path to trainning data
	CKPTPath             string // path to checkpoint
	GPURequiredPerWorker int    //
	Replicas             int    // works
}

func (kfm MPIJob) GenYaml() string {
	// TODO: impl it if needed
	return ""
}

func (kfm MPIJob) genJsonData() map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "kubeflow.org/v2beta1",
		"kind":       "MPIJob",
		"metadata": map[string]interface{}{
			"name":      kfm.Name,
			"namespace": kfm.Namespace,
		},
		"spec": map[string]interface{}{
			"launcherCreationPolicy": "AtStartup",
			"mpiImplementation":      "OpenMPI",
			"mpiReplicaSpecs": map[string]interface{}{
				"Launcher": map[string]interface{}{
					"replicas": 1,
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"command": []string{
										"mpirun",
										"-np",
										"32",
										"--allow-run-as-root",
										"-bind-to",
										"none",
										"-map-by",
										"slot",
										"-x",
										"NCCL_DEBUG=INFO",
										"-x",
										"NCCL_P2P_DISABLE=1",
										"-x",
										"LD_LIBRARY_PATH",
										"-x",
										"PATH",
										"-mca",
										"mpi_warn_on_fork",
										"0",
										"python3",
										"train_bert_ds_original.py",
										"--checkpoint_dir",
										kfm.CKPTPath,
										"--deepspeed_mpi",
										"--deepspeed",
									},
									"image":           kfm.Image,
									"imagePullPolicy": "IfNotPresent",
									"name":            "deepspeed-mpijob-container",
								},
							},
							"hostIPC": true,
						},
					},
				},
				"Worker": map[string]interface{}{
					"replicas": kfm.Replicas,
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"image":           kfm.Image,
									"imagePullPolicy": "IfNotPresent",
									"name":            "deepspeed-mpijob-container",
									"resources": map[string]interface{}{
										"limits": map[string]interface{}{
											"nvidia.com/gpu": kfm.GPURequiredPerWorker,
										},
									},
								},
							},
							"hostIPC": true,
						},
					},
				},
			},
			"runPolicy": map[string]interface{}{
				"cleanPodPolicy": "None",
				"suspend":        false,
			},
			"slotsPerWorker":   8,
			"sshAuthMountPath": "/root/.ssh",
		},
	}
}

func (kfm MPIJob) Run() error {
	return clientgo.ApplyWithJsonData(kfm.Namespace, "kubeflow.org", "v2beta1", "mpijobs", kfm.genJsonData())
}

func (kfm MPIJob) Delete() error {
	return clientgo.DeleteWithName(kfm.Namespace, "kubeflow.org", "v2beta1", "mpijobs", kfm.Name)
}

func listMPIJob(namespace string) ([]interface{}, error) {
	lst, err := clientgo.GetObjects(namespace, "kubeflow.org", "v2beta1", "mpijobs")
	if err != nil {
		return nil, err
	}
	res := []interface{}{}
	for _, item := range lst {
		res = append(res, item.Object)
	}
	return res, nil
}
