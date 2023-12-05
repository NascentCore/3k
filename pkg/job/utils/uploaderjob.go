package utils

// NO_TEST_NEEDED

import (
	"sxwl/3k/pkg/config"
)

func GenK8SJobJsonData(jobName, image, pvc, mountPath string) map[string]interface{} {
	volumeName := "modelsave-pv" //should be read twice below
	return map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "Job",
		"metadata": map[string]interface{}{
			"name":      GenModelUploaderJobName(jobName),
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
					"serviceAccountName": config.K8S_SA_NAME_FOR_MODELUPLOADER,
					"volumes": []interface{}{
						map[string]interface{}{
							"name": volumeName,
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": pvc,
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
								map[string]string{
									"name":  "DEPLOY",
									"value": config.DEPLOY,
								},
								map[string]interface{}{
									"name": config.MARKET_ACCESS_KEY,
									"valueFrom": map[string]interface{}{
										"configMapKeyRef": map[string]interface{}{
											"name": "cpod-info",
											"key":  config.MARKET_ACCESS_KEY,
										},
									},
								},
								map[string]interface{}{
									"name": config.OSS_ACCESS_KEY_ENV_NAME,
									"valueFrom": map[string]interface{}{
										"secretKeyRef": map[string]interface{}{
											"name": config.K8S_SECRET_NAME_FOR_OSS,
											"key":  config.OSS_ACCESS_KEY_ENV_NAME,
										},
									},
								},
								map[string]interface{}{
									"name": config.OSS_ACCESS_SECRET_ENV_NAME,
									"valueFrom": map[string]interface{}{
										"secretKeyRef": map[string]interface{}{
											"name": config.K8S_SECRET_NAME_FOR_OSS,
											"key":  config.OSS_ACCESS_SECRET_ENV_NAME,
										},
									},
								},
							},
							"volumeMounts": []interface{}{
								map[string]interface{}{
									"mountPath": mountPath,
									"name":      volumeName,
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
