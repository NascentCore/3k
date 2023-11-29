package generaljob

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/job/utils"
	"sxwl/3k/pkg/utils/consts"
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
	Deadline             string // 运行截止时间
}

func (gj GeneralJob) genJsonData() map[string]interface{} {
	ckptVolumeName := "ckpt-pv"
	modelSaveVolumeName := "saved-model-pv"
	dataSetVolumeName := "dataset-pv"
	pretrainModelVolumeName := "pretrain-pv"
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
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"image":           gj.Image,
							"imagePullPolicy": "IfNotPresent",
							"name":            "worker",
							"resources": map[string]interface{}{
								"limits": map[string]interface{}{
									"nvidia.com/gpu": gj.GPURequiredPerWorker,
								},
							},
							"volumeMounts": []interface{}{
								map[string]interface{}{
									"name":      dataSetVolumeName,
									"mountPath": gj.DataPath,
								},
								map[string]interface{}{
									"name":      pretrainModelVolumeName,
									"mountPath": gj.PretrainModelPath,
								},
								map[string]interface{}{
									"mountPath": gj.CKPTPath,
									"name":      ckptVolumeName,
								},
								map[string]interface{}{
									"mountPath": gj.ModelSavePath,
									"name":      modelSaveVolumeName,
								},
							},
						},
					},
					"hostIPC": true,
					"nodeSelector": map[string]interface{}{
						consts.K8S_LABEL_NV_GPU_PRODUCT: gj.GPUType,
					},
					"volumes": []interface{}{
						map[string]interface{}{
							"name": dataSetVolumeName,
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": gj.DataPVC,
								"readOnly":  true,
							},
						},
						map[string]interface{}{
							"name": pretrainModelVolumeName,
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": gj.PretrainModelPVC,
								"readOnly":  true,
							},
						},
						map[string]interface{}{
							"name": ckptVolumeName,
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": utils.GetCKPTPVCName(gj.Name),
								"readOnly":  false,
							},
						},
						map[string]interface{}{
							"name": modelSaveVolumeName,
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": utils.GetModelSavePVCName(gj.Name),
								"readOnly":  false,
							},
						},
					},
				},
			},
		},
	}
}

func (gj GeneralJob) Run() error {
	return clientgo.ApplyWithJsonData(gj.Namespace, "batch", "v1", "jobs", gj.genJsonData())
}

func (gj GeneralJob) Delete() error {
	return clientgo.DeleteK8SJob(gj.Namespace, gj.Name)
}
