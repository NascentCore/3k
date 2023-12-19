package kubeflowpytorchjob

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/job/utils"
)

// NO_TEST_NEEDED

// TODO: 根据实际情况增加或者删除一些字段
// 表示创建一个KubeFLowPytorchJob需要知道的信息.
type PytorchJob struct {
	Name                 string // will be used as metadata.name
	Namespace            string // k8s namespace
	Image                string // docker image
	DataPath             string // path to trainning data
	DataPVC              string //训练数据所在的PVC
	CKPTPath             string // path to checkpoint
	PretrainModelPath    string //预训练模型的路径
	PretrainModelPVC     string //预训练模型所在的PVC
	ModelSavePath        string //最终模型的保存路径
	GPUType              string //GPU类型
	GPURequiredPerWorker int    //每个实例所需要的GPU数量
	Command              []string
	Replicas             int    // workers
	Deadline             string // 运行截止时间
}

func (kfp PytorchJob) genJsonData() map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "kubeflow.org/v1",
		"kind":       "PyTorchJob",
		"metadata": map[string]interface{}{
			"name":      kfp.Name,
			"namespace": kfp.Namespace,
			"labels": map[string]interface{}{
				"deadline": kfp.Deadline,
			},
		},
		"spec": map[string]interface{}{
			"pytorchReplicaSpecs": map[string]interface{}{
				"Worker": map[string]interface{}{
					"replicas":      kfp.Replicas,
					"restartPolicy": "OnFailure",
					"template": utils.GenPodTemplate("pytorch", kfp.Name, kfp.Image, "IfNotPresent", kfp.GPURequiredPerWorker, kfp.GPUType,
						kfp.Command, nil, kfp.DataPVC, kfp.DataPath, kfp.PretrainModelPVC, kfp.PretrainModelPath, kfp.CKPTPath, kfp.ModelSavePath, true),
				},
			},
		},
	}
}

func (kfp PytorchJob) Run() error {
	return clientgo.ApplyWithJsonData(kfp.Namespace, "kubeflow.org", "v1", "pytorchjobs", kfp.genJsonData())
}

func (kfp PytorchJob) Delete() error {
	return clientgo.DeleteWithName(kfp.Namespace, "kubeflow.org", "v1", "pytorchjobs", kfp.Name)
}

func listPytorchJob(namespace string) ([]interface{}, error) {
	lst, err := clientgo.GetObjects(namespace, "kubeflow.org", "v1", "pytorchjobs")
	if err != nil {
		return nil, err
	}
	res := []interface{}{}
	for _, item := range lst {
		res = append(res, item.Object)
	}
	return res, nil
}
