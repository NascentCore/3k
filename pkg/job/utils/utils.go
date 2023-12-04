package utils

import "sxwl/3k/pkg/config"

func GetModelSavePVCName(jobName string) string {
	return jobName + "-modelsave"
}

func GetCKPTPVCName(jobName string) string {
	return jobName + "-ckpt"
}

func GenModelUploaderJobName(jobName string) string {
	return config.MODELUPLOADER_JOBNAME_PREFIX + jobName
}

func ParseJobNameFromModelUploader(uploaderJob string) string {
	prefix := config.MODELUPLOADER_JOBNAME_PREFIX
	if len(uploaderJob) > len(prefix) && uploaderJob[:len(prefix)] == prefix {
		return uploaderJob[len(prefix):]
	}
	return ""
}
