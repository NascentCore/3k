package generaljob

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/job/utils"
)

// NO_TEST_NEEDED

// TODO: 根据实际情况增加或者删除一些字段
// 表示创建一个KubeFLowGeneralJob需要知道的信息.
type GeneralJob struct {
	Name                 string // will be used as metadata.name
	Namespace            string // k8s namespace
	Image                string // docker image
	DataPath             string // path to trainning data
	DataPVC              string //训练数据所在的PVC
	CKPTPath             string // path to checkpoint
	PretrainModelPath    string //预训练模型的路径
	PretrainModelPVC     string //预训练模型所在的PVC
	ModelSavePath        string //最终模型的保存路径
	GPUType              string
	GPURequiredPerWorker int //
	Command              []string
	Envs                 map[string]string //环境变量列表
	Deadline             string            // 运行截止时间
}

func (gj GeneralJob) genJsonData() map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "Job",
		"metadata": map[string]interface{}{
			"name":      gj.Name,
			"namespace": gj.Namespace,
			"labels": map[string]interface{}{
				"jobtype":  "generaljob", // with this label means this job is generaljob
				"deadline": gj.Deadline,
			},
		},
		"spec": map[string]interface{}{
			"backoffLimit":   10,
			"completionMode": "NonIndexed",
			"completions":    1,
			"parallelism":    1,
			"suspend":        false,
			"template": utils.GenPodTemplate("generaljob", gj.Name, gj.Image, "IfNotPresent", gj.GPURequiredPerWorker, gj.GPUType,
				gj.Command, gj.Envs, gj.DataPVC, gj.DataPath, gj.PretrainModelPVC, gj.PretrainModelPath, gj.CKPTPath, gj.ModelSavePath, false),
		},
	}
}

func (gj GeneralJob) Run() error {
	return clientgo.ApplyWithJsonData(gj.Namespace, "batch", "v1", "jobs", gj.genJsonData())
}

func (gj GeneralJob) Delete() error {
	return clientgo.DeleteK8SJob(gj.Namespace, gj.Name)
}
