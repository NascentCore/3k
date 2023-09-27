package kubeflowmpijob

import (
	clientgo "sxwl/3k/manager/pkg/cluster/client-go"
)

/**
* example of MPIJOB
{
    "apiVersion": "kubeflow.org/v2beta1",
    "kind": "MPIJob",
    "metadata": {
        "annotations": {
            "kubectl.kubernetes.io/last-applied-configuration": "{\"apiVersion\":\"kubeflow.org/v2beta1\",\"kind\":\"MPIJob\",\"metadata\":{\"annotations\":{},\"name\":\"dp-mpijob-bert\",\"namespace\":\"default\"},\"spec\":{\"mpiReplicaSpecs\":{\"Launcher\":{\"replicas\":1,\"template\":{\"spec\":{\"containers\":[{\"command\":[\"mpirun\",\"-np\",\"32\",\"--allow-run-as-root\",\"-bind-to\",\"none\",\"-map-by\",\"slot\",\"-x\",\"NCCL_DEBUG=INFO\",\"-x\",\"NCCL_P2P_DISABLE=1\",\"-x\",\"LD_LIBRARY_PATH\",\"-x\",\"PATH\",\"-mca\",\"mpi_warn_on_fork\",\"0\",\"python3\",\"train_bert_ds_original.py\",\"--checkpoint_dir\",\"./ds_experiments\",\"--deepspeed_mpi\",\"--deepspeed\"],\"image\":\"swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim\",\"imagePullPolicy\":\"IfNotPresent\",\"name\":\"deepspeed-mpijob-container\"}],\"hostIPC\":true}}},\"Worker\":{\"replicas\":4,\"template\":{\"spec\":{\"containers\":[{\"image\":\"swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim\",\"imagePullPolicy\":\"IfNotPresent\",\"name\":\"deepspeed-mpijob-container\",\"resources\":{\"limits\":{\"nvidia.com/gpu\":8}}}],\"hostIPC\":true}}}},\"runPolicy\":{\"cleanPodPolicy\":\"None\"},\"slotsPerWorker\":8}}\n"
        },
        "creationTimestamp": "2023-09-26T02:56:41Z",
        "generation": 1,
        "name": "dp-mpijob-bert",
        "namespace": "default",
        "resourceVersion": "1145161",
        "uid": "94d3af95-0e61-46bb-a6ff-648de4ea89f9"
    },
    "spec": {
        "launcherCreationPolicy": "AtStartup",
        "mpiImplementation": "OpenMPI",
        "mpiReplicaSpecs": {
            "Launcher": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "command": [
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
                                    "./ds_experiments",
                                    "--deepspeed_mpi",
                                    "--deepspeed"
                                ],
                                "image": "swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim",
                                "imagePullPolicy": "IfNotPresent",
                                "name": "deepspeed-mpijob-container"
                            }
                        ],
                        "hostIPC": true
                    }
                }
            },
            "Worker": {
                "replicas": 4,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "image": "swr.cn-east-3.myhuaweicloud.com/sxwl/bert:v10jim",
                                "imagePullPolicy": "IfNotPresent",
                                "name": "deepspeed-mpijob-container",
                                "resources": {
                                    "limits": {
                                        "nvidia.com/gpu": 8
                                    }
                                }
                            }
                        ],
                        "hostIPC": true
                    }
                }
            }
        },
        "runPolicy": {
            "cleanPodPolicy": "None",
            "suspend": false
        },
        "slotsPerWorker": 8,
        "sshAuthMountPath": "/root/.ssh"
    },
    "status": {
        "completionTime": "2023-09-26T04:03:07Z",
        "conditions": [
            {
                "lastTransitionTime": "2023-09-26T02:56:44Z",
                "lastUpdateTime": "2023-09-26T02:56:44Z",
                "message": "MPIJob default/dp-mpijob-bert is created.",
                "reason": "MPIJobCreated",
                "status": "True",
                "type": "Created"
            },
            {
                "lastTransitionTime": "2023-09-26T04:03:07Z",
                "lastUpdateTime": "2023-09-26T04:03:07Z",
                "message": "Job has reached the specified backoff limit",
                "reason": "BackoffLimitExceeded",
                "status": "True",
                "type": "Failed"
            },
            {
                "lastTransitionTime": "2023-09-26T04:03:07Z",
                "lastUpdateTime": "2023-09-26T04:03:07Z",
                "message": "MPIJob default/dp-mpijob-bert is running.",
                "reason": "MPIJobRunning",
                "status": "True",
                "type": "Running"
            }
        ],
        "replicaStatuses": {
            "Launcher": {
                "failed": 1
            },
            "Worker": {}
        },
        "startTime": "2023-09-26T02:56:44Z"
    }
}
*/

type KubeFlowMPIJob struct {
	Name        string
	Namespace   string
	Image       string
	DataPath    string
	CKPTPath    string
	GPURequired int
	Replicas    int
}

func (kfm KubeFlowMPIJob) GenYaml() string {
	//TODO
	return ""
}

func (kfm KubeFlowMPIJob) genJsonData() map[string]interface{} {
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
											"nvidia.com/gpu": kfm.GPURequired,
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

func (kfm KubeFlowMPIJob) Run() error {
	return clientgo.ApplyWithJsonData(kfm.Namespace, "kubeflow.org", "v2beta1", "mpijobs", kfm.genJsonData())
}

func (kfm KubeFlowMPIJob) Delete() error {
	//TODO
	return nil
}

func (kfm KubeFlowMPIJob) Get() (map[string]interface{}, error) {
	//TODO
	return nil, nil
}

func ListMPIJob() error {
	//TODO
	return nil
}
