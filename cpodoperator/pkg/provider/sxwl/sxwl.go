package sxwl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"

	v1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
)

var _ Scheduler = &sxwl{}

type sxwl struct {
	httpClient *http.Client
	baseURL    string
	accessKey  string
	identity   string
}

type GetJobResponse struct {
	TrainningJobs  []PortalTrainningJob  `json:"job_list"`
	InferenceJobs  []PortalInferenceJob  `json:"inference_service_list"`
	JupyterLabJobs []PortalJupyterLabJob `json:"jupyter_lab_list"`
}

// GetAssignedJobList implements Scheduler.
func (s *sxwl) GetAssignedJobList() ([]PortalTrainningJob, []PortalInferenceJob, []PortalJupyterLabJob, []UserID, error) {
	urlStr, err := url.JoinPath(s.baseURL, v1beta1.URLPATH_FETCH_JOB)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	req, err := http.NewRequest(http.MethodGet, urlStr, nil)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	q := req.URL.Query()
	q.Add("cpodid", s.identity)
	req.URL.RawQuery = q.Encode()
	req.Header.Add("Authorization", "Bearer "+s.accessKey)
	req.Header.Add("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, nil, nil, nil, fmt.Errorf("httpcode(%d) is not 200 , resp body: %s", resp.StatusCode, string(body))
	}

	var res GetJobResponse
	if err = json.Unmarshal(body, &res); err != nil {
		return nil, nil, nil, nil, err
	}
	userIDs := []UserID{}
	for _, v := range res.TrainningJobs {
		userIDs = append(userIDs, UserID(strconv.Itoa(int(v.UserID))))
	}

	for _, v := range res.InferenceJobs {
		exist := false
		for _, userID := range userIDs {
			if userID == UserID(v.UserID) {
				exist = true
				break
			}
		}
		if !exist {
			userIDs = append(userIDs, UserID(strconv.Itoa(int(v.UserID))))
		}
	}
	for _, v := range res.JupyterLabJobs {
		exist := false
		for _, userID := range userIDs {
			if userID == UserID(v.UserID) {
				exist = true
				break
			}
		}
		if !exist {
			userIDs = append(userIDs, UserID(strconv.Itoa(int(v.UserID))))
		}
	}
	return res.TrainningJobs, res.InferenceJobs, res.JupyterLabJobs, userIDs, nil
}

func (s *sxwl) HeartBeat(payload HeartBeatPayload) error {
	urlStr, err := url.JoinPath(s.baseURL, v1beta1.URLPATH_UPLOAD_CPOD_STATUS)
	if err != nil {
		return err
	}
	reqBytes, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, urlStr, bytes.NewBuffer(reqBytes))
	if err != nil {
		return err
	}
	req.Header.Add("Authorization", "Bearer "+s.accessKey)
	req.Header.Add("Content-Type", "application/json")
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respData, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("statuscode(%d) != 200 and read body err : %v", resp.StatusCode, err)
		}
		fmt.Println(string(respData))
		return fmt.Errorf("statuscode(%d) != 200", resp.StatusCode)
	}
	return nil
}
