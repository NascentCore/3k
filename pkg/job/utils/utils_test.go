package utils

import (
	"sxwl/3k/pkg/config"
	"testing"
)

func TestParseJobNameFromModelUploader(t *testing.T) {
	jobName := "aaa123"
	modeluploaderJobName := config.MODELUPLOADER_JOBNAME_PREFIX + jobName
	jobName_ := ParseJobNameFromModelUploader(modeluploaderJobName)
	if jobName != jobName_ {
		t.Error("error result")
	}

}
