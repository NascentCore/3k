package modeluploader

func GenK8SJobJsonData(jobName, image, pvc, mountPath string, cmd []interface{}) map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "batch/v1",
		"kind":       "Job",
		"metadata": map[string]interface{}{
			"name":      jobName,
			"namespace": "cpod",
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
							"command":                  cmd,
							"image":                    image,
							"imagePullPolicy":          "IfNotPresent",
							"name":                     jobName,
							"resources":                map[string]interface{}{},
							"terminationMessagePath":   "/dev/termination-log",
							"terminationMessagePolicy": "File",
							"volumeMounts": []interface{}{
								map[string]interface{}{
									"mountPath": mountPath,
									"name":      "modelpvc",
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
					"volumes": []interface{}{
						map[string]interface{}{
							"name": "modelpvc",
							"persistentVolumeClaim": map[string]interface{}{
								"claimName": pvc,
							},
						},
					},
				},
			},
		},
	}
}
