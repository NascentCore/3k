package communication

import (
	"sxwl/3k/pkg/config"
	"testing"
)

func Test_validatePath(t *testing.T) {
	badCases := []string{"", ".", "./123", "../abc", "abc"}
	for _, badCase := range badCases {
		if err := validatePath(badCase); err == nil {
			t.Error(badCase)
		}
	}
	goodCases := []string{"/aa", "/aa/bb/", "/aa/bb"}
	for _, goodCase := range goodCases {
		if err := validatePath(goodCase); err != nil {
			t.Error(goodCase)
		}
	}
}

func TestValidateJob(t *testing.T) {
	j := RawJobDataItem{
		CkptPath:          "",
		CkptVol:           0,
		Command:           "",
		Envs:              map[string]string{},
		DatasetPath:       "",
		DatasetName:       "",
		GpuNumber:         1,
		GpuType:           "3090",
		HfURL:             "",
		ImagePath:         "image1",
		JobID:             0,
		JobName:           "job1",
		JobType:           config.PORTAL_JOBTYPE_MPI,
		ModelPath:         "/model",
		ModelVol:          100,
		PretrainModelName: "",
		PretrainModelPath: "",
		StopTime:          0,
		StopType:          0,
	}
	datasets := map[string]struct{}{"dataset1": {}}
	models := map[string]struct{}{"model1": {}}
	if err := ValidateJob(j, datasets, models); err != nil {
		t.Error(err)
	}
	j.DatasetName = "dataset1"
	j.DatasetPath = "/data"
	if err := ValidateJob(j, datasets, models); err != nil {
		t.Error(err)
	}
	j.CkptVol = 100
	if err := ValidateJob(j, datasets, models); err == nil {
		t.Error()
	}
	j.CkptPath = "/ckpt"
	if err := ValidateJob(j, datasets, models); err != nil {
		t.Error(err)
	}
}
