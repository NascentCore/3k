package utils

// NO_TEST_NEEDED

func GetModelSavePVCName(jobName string) string {
	return jobName + "-modelsave"
}

func GetCKPTPVCName(jobName string) string {
	return jobName + "-ckpt"
}
