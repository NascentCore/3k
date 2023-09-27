// NO_TEST_NEEDED
package db

import "gorm.io/gorm"

type JobScheduler struct {
	gorm.Model
	Id        int    `json:"id,omitempty"`
	JobId     string `json:"job_id"`
	CpodJobId string `json:"cpod_job_id"`
	CpodId    string `json:"cpod_job_id"`
	State     int    `json:"state"`
	JobUrl    string `json:"job_url"`
}

func (j *JobScheduler) CreateJob(db *gorm.DB) (string, error) {
	results := db.Table("job_scheduler").Create(j)
	if results.Error != nil {
		return results.Error.Error(), results.Error
	}
	return "", nil
}

func (j *JobScheduler) UpdateJob(db *gorm.DB) (string, error) {
	results := db.Table("job_scheduler").Where("job_id=?", j.JobId).Update("job_url", j.JobUrl)
	if results.Error != nil {
		return results.Error.Error(), results.Error
	}
	return "", nil
}
