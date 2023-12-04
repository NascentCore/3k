package job

// NO_TEST_NEEDED

import (
	clientgo "sxwl/3k/pkg/cluster/client-go"
	"sxwl/3k/pkg/config"
	generaljob "sxwl/3k/pkg/job/general-job"
	kubeflowmpijob "sxwl/3k/pkg/job/kubeflow-mpijob"
	kubeflowpytorchjob "sxwl/3k/pkg/job/kubeflow-pytorchjob"
	"sxwl/3k/pkg/job/state"
	"sxwl/3k/pkg/job/utils"
	"sxwl/3k/pkg/log"

	v1 "k8s.io/api/batch/v1"
)

// if job is succeeded or deleted and uploadjob is not complete ,
// jobstates is modeluploading
// if job is succeeded or deleted and uploadjob is complete ,
// status is modeluploaded
// just for status upload
func GetJobStatesWithUploaderInfo() ([]state.State, error) {
	states, err := GetJobStates()
	if err != nil {
		return []state.State{}, err
	}
	jobStateMap := map[string]state.State{}
	for _, state := range states {
		jobStateMap[state.Name] = state
	}
	//get uploader job list
	uploaderJobs, err := GetUploaderJobs(config.CPOD_NAMESPACE)
	if err != nil {
		return []state.State{}, err
	}
	res := []state.State{}
	jobAdded := map[string]struct{}{}
	for _, uploaderJob := range uploaderJobs {
		jobName := utils.ParseJobNameFromModelUploader(uploaderJob.Name)
		if jobState, ok := jobStateMap[jobName]; ok {
			log.SLogger.Infow("job is found", "jobName", jobName)
			if jobState.JobStatus == state.JobStatusSucceed {
				log.SLogger.Infow("job status is succeed", "jobName", jobName)
				s := jobState
				if isJobComplete(uploaderJob) {
					s.JobStatus = state.JobStatusModelUploaded
				}
				res = append(res, s)
				jobAdded[s.Name] = struct{}{}
			}
		} else {
			//job is deleted
			log.SLogger.Infow("job is deleted", "jobName", jobName)
			s := state.State{
				Name:      jobName,
				Namespace: config.CPOD_NAMESPACE,
			}
			if isJobComplete(uploaderJob) {
				log.SLogger.Infow("uploader is completed", "jobName", jobName)
				s.JobStatus = state.JobStatusModelUploaded
			} else {
				log.SLogger.Infow("uploader is running", "jobName", jobName)
				s.JobStatus = state.JobStatusModelUploading
			}
			res = append(res, s)
			jobAdded[s.Name] = struct{}{}
		}
	}
	// add other items
	for _, state := range states {
		if _, ok := jobAdded[state.Name]; !ok {
			res = append(res, state)
		}
	}
	return res, nil
}

func isJobComplete(j *v1.Job) bool {
	for _, cond := range j.Status.Conditions {
		if string(cond.Type) == "Complete" && string(cond.Status) == "True" {
			return true
		}
	}
	return false
}

func GetUploaderJobs(namespace string) ([]*v1.Job, error) {
	data, err := clientgo.GetK8SJobs(namespace)
	if err != nil {
		return []*v1.Job{}, err
	}
	//从Data中提取State信息
	res := []*v1.Job{}
	for _, item := range data.Items {
		// filter with name
		if utils.ParseJobNameFromModelUploader(item.Name) == "" {
			continue
		}
		res = append(res, &item)
	}
	return res, nil
}

func GetJobStates() ([]state.State, error) {
	res := []state.State{}
	//对于不同类型的任务，分别获取其任务状态，加入到结果中
	// KubeflowMPI
	if data, err := kubeflowmpijob.GetStates(config.CPOD_NAMESPACE); err == nil {
		res = append(res, data...)
	} else {
		log.SLogger.Errorw("err when get mpijob states", "error", err)
		return []state.State{}, err
	}
	// KubeflowPytorch
	if data, err := kubeflowpytorchjob.GetStates(config.CPOD_NAMESPACE); err == nil {
		res = append(res, data...)
	} else {
		log.SLogger.Errorw("err when get pytorchjob states", "error", err)
		return []state.State{}, err
	}
	// GeneralJob
	if data, err := generaljob.GetStates(config.CPOD_NAMESPACE); err == nil {
		res = append(res, data...)
	} else {
		log.SLogger.Errorw("err when get generaljob states", "error", err)
		return []state.State{}, err
	}
	return res, nil
}
