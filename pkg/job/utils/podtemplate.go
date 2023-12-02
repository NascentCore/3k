package utils

import "sxwl/3k/pkg/utils/consts"

func GenPodTemplate(jobName, image, imagePolicy string, gpus int, gpuType string, command []string,
	datasetPVC, datasetPath, pretrainModelPVC, pretrainModelPath,
	ckptPath, modelSavePath string, withSHM bool) map[string]interface{} {

	ckptVolumeName := "ckpt-pv"
	modelSaveVolumeName := "saved-model-pv"
	dataSetVolumeName := "dataset-pv"
	pretrainModelVolumeName := "pretrain-pv"
	// modelsaveVolume will be mount
	volumes := []interface{}{
		map[string]interface{}{
			"name": modelSaveVolumeName,
			"persistentVolumeClaim": map[string]interface{}{
				"claimName": GetModelSavePVCName(jobName),
				"readOnly":  false,
			},
		},
	}
	volumeMounts := []interface{}{
		map[string]interface{}{
			"mountPath": modelSavePath,
			"name":      modelSaveVolumeName,
		},
	}
	// if ckptPath == ""   no ckptvolume mount
	if ckptPath != "" {
		volumes = append(volumes, map[string]interface{}{
			"name": ckptVolumeName,
			"persistentVolumeClaim": map[string]interface{}{
				"claimName": GetCKPTPVCName(jobName),
				"readOnly":  false,
			},
		})
		volumeMounts = append(volumeMounts, map[string]interface{}{
			"mountPath": ckptPath,
			"name":      ckptVolumeName,
		})
	}
	// if datasetPVC == ""   no dataset mount
	if datasetPVC != "" {
		volumes = append(volumes, map[string]interface{}{
			"name": dataSetVolumeName,
			"persistentVolumeClaim": map[string]interface{}{
				"claimName": datasetPVC,
				"readOnly":  true,
			},
		})
		volumeMounts = append(volumeMounts, map[string]interface{}{
			"name":      dataSetVolumeName,
			"mountPath": datasetPath,
		})
	}
	// if pretrainModelPVC == ""   no pretrainmound mount
	if pretrainModelPVC != "" {
		volumes = append(volumes, map[string]interface{}{
			"name": pretrainModelVolumeName,
			"persistentVolumeClaim": map[string]interface{}{
				"claimName": pretrainModelPVC,
				"readOnly":  true,
			},
		})
		volumeMounts = append(volumeMounts, map[string]interface{}{
			"name":      pretrainModelVolumeName,
			"mountPath": pretrainModelPath,
		})
	}
	if withSHM {
		volumes = append(volumes, map[string]interface{}{
			"name": "shm",
			"emptyDir": map[string]interface{}{
				"medium":    "Memory",
				"sizeLimit": "5120Mi",
			},
		})
		volumeMounts = append(volumeMounts, map[string]interface{}{
			"name":      "shm",
			"mountPath": "/dev/shm",
		})
	}

	container := map[string]interface{}{
		"name":            "container1",
		"image":           image,
		"imagePullPolicy": imagePolicy,
		"resources": map[string]interface{}{
			"limits": map[string]interface{}{
				"nvidia.com/gpu": gpus,
			},
		},
		"volumeMounts": volumeMounts,
	}
	// if command is set
	if command != nil && len(command) != 0 {
		container["command"] = command
	}

	return map[string]interface{}{
		"spec": map[string]interface{}{
			"restartPolicy": "OnFailure",
			"containers": []interface{}{
				container,
			},
			"nodeSelector": map[string]interface{}{
				consts.K8S_LABEL_NV_GPU_PRODUCT: gpuType,
			},
			"volumes": volumes,
		},
	}
}
