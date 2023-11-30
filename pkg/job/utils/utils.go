package utils

// NO_TEST_NEEDED

func GetModelSavePVCName(jobName string) string {
	return jobName + "-modelsave"
}

func GetCKPTPVCName(jobName string) string {
	return jobName + "-ckpt"
}

func GenModelUploaderJobName(jobName string) string {
	return "modeluploader-" + jobName
}

func ParseJobNameFromModelUploader(uploaderJob string) string {
	prefix := "modeluploader-"
	if len(uploaderJob) > len(prefix) && uploaderJob[:len(prefix)] == prefix {
		return uploaderJob[len(prefix):]
	}
	return ""
}
