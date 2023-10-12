package logic

import "testing"

// TODO(yzhao): "github.com/stretchr/testify/mock" Use this to update the test

func TestCpodJobLogical(t *testing.T) {
	jobBody := `{
	"job_id": "87e06ce2-1321-4a1d-acbf-e646ef4fe161",
	"cpod_job_id": "14fd7e86-3fd8-4df1-93b1-de4e0d484955",
	"state": 0,
	"job_url": ""
	}`
	_, err := CpodJobLogical([]byte(jobBody))
	if err != nil {
		t.Error(err)
	}
}

func TestCpodJobResultLogical(t *testing.T) {
	jobBody := `{
	"job_id": "87e06ce2-1321-4a1d-acbf-e646ef4fe161",
	"cpod_job_id": "14fd7e86-3fd8-4df1-93b1-de4e0d484955",
	"state": 0,
	"job_url": "sxwl.croe.ai"
	}`
	_, err := CpodJobResultLogical([]byte(jobBody))
	if err != nil {
		t.Error(err)
	}
}

func TestCpodResourceLogical(t *testing.T) {
	jobBody := `{
	"cpod_id": "87e06ce2-1321-4a1d-acbf-e646ef4fe161",
	"gpu_total": 0.1,
	"gpu_used": 0.1,
	"gpu_free": 0.1
    }`
	_, err := CpodResourceLogical([]byte(jobBody))
	if err != nil {
		t.Error(err)
	}
}
