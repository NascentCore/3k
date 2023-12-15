package communication

import (
	"fmt"
	"sxwl/3k/pkg/config"
)

func validatePath(p string) error {
	return nil
}

// if validate ok , return nil
func ValidateJob(j RawJobDataItem, datasets, models map[string]struct{}) error {
	if j.CkptVol < 0 {
		return fmt.Errorf("ckptvol(%d) < 0", j.CkptVol)
	} else if j.CkptVol == 0 {
		if j.CkptPath != "" {
			return fmt.Errorf("ckpt path is set but vol is 0")
		}
	} else {
		if j.CkptVol <= 10 {
			return fmt.Errorf("ckptvol(%d) too small", j.CkptVol)
		}
		if j.CkptVol >= 100*1024 {
			return fmt.Errorf("ckptvol(%d) too large", j.CkptVol)
		}
		if err := validatePath(j.CkptPath); err != nil {
			return err
		}
	}

	if j.ModelVol <= 10 {
		return fmt.Errorf("modelsavevol(%d) too small", j.ModelVol)
	}
	if j.ModelVol >= 100*1024 {
		return fmt.Errorf("modelsavevol(%d) too large", j.ModelVol)
	}
	if err := validatePath(j.ModelPath); err != nil {
		return err
	}

	if j.DatasetName != "" {
		if _, ok := datasets[j.DatasetName]; !ok {
			return fmt.Errorf("dataset(%s) not exist", j.DatasetName)
		}
		if err := validatePath(j.DatasetPath); err != nil {
			return err
		}
	} else {
		if j.DatasetPath != "" {
			return fmt.Errorf("specified path but no dataset")
		}
	}

	if j.PretrainModelName != "" {
		if _, ok := models[j.PretrainModelName]; !ok {
			return fmt.Errorf("model(%s) not exist", j.PretrainModelName)
		}
		if err := validatePath(j.PretrainModelPath); err != nil {
			return err
		}
	} else {
		if j.PretrainModelPath != "" {
			return fmt.Errorf("specified path but no model")
		}
	}
	if j.JobType != config.PORTAL_JOBTYPE_GENERAL && j.JobType != config.PORTAL_JOBTYPE_MPI &&
		j.JobType != config.PORTAL_JOBTYPE_PYTORCH && j.JobType != config.PORTAL_JOBTYPE_TENSORFLOW {
		return fmt.Errorf("invalid jobtype(%s)", j.JobType)
	}
	return nil
}
