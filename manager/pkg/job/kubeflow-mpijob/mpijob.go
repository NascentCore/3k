package kubeflowmpijob

import (
	"fmt"
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
	PretrainModelPath    string //预训练模型的路径
	ModelSavePath        string //最终模型的保存路径
	GPUType              string
	GPURequiredPerWorker int //
	Replicas             int // works
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
			"launcherCreationPolicy": "WaitForWorkersReady",
			"mpiImplementation":      "OpenMPI",
			"mpiReplicaSpecs": map[string]interface{}{
				"Launcher": map[string]interface{}{
					"replicas": 1,
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"command": []interface{}{
										"mpirun",
										"-np",
										fmt.Sprintf("%d", kfm.GPURequiredPerWorker*kfm.Replicas),
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
										"train_bert_ds.py",
										"--checkpoint_dir",
										"ds-experiments",
										"--dataset_dir",
										"dataset1/wikitext",
										"--num_iterations=50",
									},
									"image":           kfm.Image,
									"imagePullPolicy": "Always",
									"name":            "bert-launcher",
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
									"imagePullPolicy": "Always",
									"name":            "bert-ds-worker",
									"resources": map[string]interface{}{
										"limits": map[string]interface{}{
											"nvidia.com/gpu": kfm.GPURequiredPerWorker,
										},
									},
									"volumeMounts": []interface{}{
										map[string]interface{}{
											"mountPath": "/workspace/dataset1",
											"name":      "dataset1",
										},
										map[string]interface{}{
											"mountPath": "/workspace/ds-experiments",
											"name":      "ckpt-pv",
										},
										map[string]interface{}{
											"mountPath": "/workspace/saved-model",
											"name":      "saved-model-pv",
										},
									},
								},
							},
							"hostIPC": true,
							"nodeSelector": map[string]interface{}{
								"nvidia.com/gpu.product": kfm.GPUType,
							},
							"volumes": []interface{}{
								map[string]interface{}{
									"cephfs": map[string]interface{}{
										"monitors": []interface{}{
											"10.233.33.169:6789",
										},
										"path":     "/readonly/hf/dataset",
										"readOnly": true,
										"secretRef": map[string]interface{}{
											"name": "ceph-secret",
										},
										"user": "admin",
									},
									"name": "dataset1",
								},
								map[string]interface{}{
									"cephfs": map[string]interface{}{
										"monitors": []interface{}{
											"10.233.33.169:6789",
										},
										"path":     "/readwrite/mvp-ckpt",
										"readOnly": false,
										"secretRef": map[string]interface{}{
											"name": "ceph-secret",
										},
										"user": "admin",
									},
									"name": "ckpt-pv",
								},
								map[string]interface{}{
									"cephfs": map[string]interface{}{
										"monitors": []interface{}{
											"10.233.33.169:6789",
										},
										"path":     "/readwrite/mvp-saved-model",
										"readOnly": false,
										"secretRef": map[string]interface{}{
											"name": "ceph-secret",
										},
										"user": "admin",
									},
									"name": "saved-model-pv",
								},
							},
						},
					},
				},
			},
			"runPolicy": map[string]interface{}{
				"cleanPodPolicy": "Running",
				"suspend":        false,
			},
			"slotsPerWorker":   kfm.GPURequiredPerWorker,
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
