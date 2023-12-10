package kubeflowmpijob

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/job/utils"
)

// NO_TEST_NEEDED

// TODO: 根据实际情况增加或者删除一些字段
// 表示创建一个KubeFLowMPIJob需要知道的信息.
type MPIJob struct {
	Name                     string // will be used as metadata.name
	Namespace                string // k8s namespace
	Image                    string // docker image
	DataPath                 string // path to trainning data
	DataPVC                  string //训练数据所在的PVC
	CKPTPath                 string // path to checkpoint
	PretrainModelPath        string //预训练模型的路径
	PretrainModelPVC         string //预训练模型所在的PVC
	ModelSavePath            string //最终模型的保存路径
	GPUType                  string
	GPURequiredPerWorker     int //
	Command                  []string
	Replicas                 int    // works
	Deadline                 string // 运行截止时间
	ExecutionDurationSeconds string
}

func (kfm MPIJob) genJsonData() map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "kubeflow.org/v2beta1",
		"kind":       "MPIJob",
		"metadata": map[string]interface{}{
			"name":      kfm.Name,
			"namespace": kfm.Namespace,
			"labels": map[string]interface{}{
				"deadline":                 kfm.Deadline,
				"executionDurationSeconds": kfm.ExecutionDurationSeconds,
			},
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
									"command":         kfm.Command,
									"image":           kfm.Image,
									"imagePullPolicy": "IfNotPresent",
									"name":            "launcher",
								},
							},
							"hostIPC": true,
						},
					},
				},
				"Worker": map[string]interface{}{
					"replicas": kfm.Replicas,
					"template": utils.GenPodTemplate(kfm.Name, kfm.Image, "IfNotPresent", kfm.GPURequiredPerWorker, kfm.GPUType,
						nil, nil, kfm.DataPVC, kfm.DataPath, kfm.PretrainModelPVC, kfm.PretrainModelPath, kfm.CKPTPath, kfm.ModelSavePath, false),
				},
			},
			"runPolicy": map[string]interface{}{
				"cleanPodPolicy": "Running",
				"suspend":        false,
				//see https://www.kubeflow.org/docs/components/training/mpi/#scheduling-policy
				"schedulingPolicy": map[string]interface{}{
					"minAvailable": kfm.Replicas,
				},
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
