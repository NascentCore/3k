package modeluploader

import "sxwl/3k/manager/pkg/config"

func GenK8SJobJsonData(jobName, image, pvc, mountPath string) map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "Job",
		"metadata": map[string]interface{}{
			"name":      jobName,
			"namespace": config.CPOD_NAMESPACE,
		},
		"spec": map[string]interface{}{
			"backoffLimit":   10,
			"completionMode": "NonIndexed",
			"completions":    1,
			"parallelism":    1,
			"suspend":        false,
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"serviceAccountName": "sa-modeluploader",
					"volumes": []interface{}{
						map[string]interface{}{
							"name": "modelsave-pv",
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": pvc, //should be saved-model in mvp
								"readOnly":  false,
							},
						},
					},
					"containers": []interface{}{
						map[string]interface{}{
							"command": []interface{}{
								"./modeluploadjob",
								jobName,
								config.OSS_BUCKET,
							},
							"name":            "uploadjob",
							"image":           image,
							"imagePullPolicy": "Always",
							"env": []interface{}{
								map[string]interface{}{
									"name": "AK",
									"valueFrom": map[string]interface{}{
										"secretKeyRef": map[string]interface{}{
											"name": "akas4oss",
											"key":  "AK",
										},
									},
								},
								map[string]interface{}{
									"name": "AS",
									"valueFrom": map[string]interface{}{
										"secretKeyRef": map[string]interface{}{
											"name": "akas4oss",
											"key":  "AS",
										},
									},
								},
							},
							"volumeMounts": []interface{}{
								map[string]interface{}{
									"mountPath": mountPath,
									"name":      "modelsave-pv",
								},
							},
						},
					},
					"dnsPolicy":                     "ClusterFirst",
					"restartPolicy":                 "OnFailure",
					"schedulerName":                 "default-scheduler",
					"securityContext":               map[string]interface{}{},
					"terminationGracePeriodSeconds": 30,
					"tolerations": []interface{}{
						map[string]interface{}{
							"effect":            "NoExecute",
							"key":               "node.kubernetes.io/not-ready",
							"operator":          "Exists",
							"tolerationSeconds": 300,
						},
						map[string]interface{}{
							"effect":            "NoExecute",
							"key":               "node.kubernetes.io/unreachable",
							"operator":          "Exists",
							"tolerationSeconds": 300,
						},
					},
				},
			},
		},
	}
}
