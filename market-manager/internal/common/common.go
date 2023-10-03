package common

// NO_TEST_NEEDED

type Response struct {
	Code int         `json:"code"`
	Msg  string      `json:"msg"`
	Data interface{} `json:"data"`
}

type CodJobInfo struct {
	JobId     string `json:"job_id"`
	CpodJobId string `json:"cpod_job_id"`
	CpodId    string `json:"cpod_job_id"`
	State     int    `json:"state"`
	JobUrl    string `json:"job_url"`
}

type CpodResourceInfos struct {
	Cpods []CpodResourceInfo `json:"cpods"`
}

type CpodResourceInfo struct {
	GroupId  string  `json:"group_id,omitempty"`
	CpodId   string  `json:"cpod_id"`
	GpuTotal float32 `json:"gpu_total"`
	GpuUsed  float32 `json:"gpu_used"`
	GpuFree  float32 `json:"gpu_free"`
}
