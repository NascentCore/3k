package modelhub

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"time"
)

type modelscope struct {
	baseURL    string
	httpClient *http.Client
}

type ModeScopeFilesResp struct {
	Code int                     `json:"Code"`
	Data *ModeScopeFilesRespData `json:"Data,omitempty"`
}

type ModeScopeFilesRespData struct {
	Files []ModeScopeFilesRespDataFile `json:"Files"`
}

type ModeScopeFilesRespDataFile struct {
	CommitMessage string `json:"CommitMessage,omitempty"`
	CommittedDate int    `json:"CommittedDate,omitempty"`
	CommitterName string `json:"CommitterName,omitempty"`
	InCheck       bool   `json:"InCheck,omitempty"`
	IsLFS         bool   `json:"IsLFS,omitempty"`
	Mode          string `json:"Mode,omitempty"`
	Name          string `json:"Name,omitempty"`
	Path          string `json:"Path,omitempty"`
	Revision      string `json:"Revision,omitempty"`
	Sha256        string `json:"Sha256,omitempty"`
	Size          int    `json:"Size,omitempty"`
	Type          string `json:"Type,omitempty"`
}

func (m *modelscope) ModelInformation(modelID, revision string) (interface{}, error) {
	targetURL, err := url.JoinPath(m.baseURL, "/api/v1/models/", modelID)
	if err != nil {
		return nil, err
	}
	if revision != "" {
		targetURL, err = url.JoinPath(targetURL, fmt.Sprintf("?Revision=%v", revision))
		if err != nil {
			return nil, err
		}
	}
	resp, err := m.httpClient.Get(targetURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil, nil
	}
	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrModelNotFound
	}
	return nil, fmt.Errorf("failed to get model information, http StatusCode is %v", resp.StatusCode)
}

// ModelSize get modelscope model size
// https://github.com/modelscope/modelscope/blob/b21afc3424a1049c91d2581e3b3294ba0a96b9e7/modelscope/hub/api.py#L541
func (m *modelscope) ModelSize(modelID, revision string) (int64, error) {
	targetURLStr, err := url.JoinPath(m.baseURL, "/api/v1/models/", modelID, "repo/files")
	if err != nil {
		return 0, err
	}
	targetURL, err := url.Parse(targetURLStr)
	if err != nil {
		return 0, err
	}
	targetURL.Query().Add("Recursive", "true")
	if revision != "" {
		targetURL.Query().Add("Revision", revision)
	}
	resp, err := m.httpClient.Get(targetURL.String())
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusNotFound {
			return 0, ErrModelNotFound
		}
		return 0, fmt.Errorf("failed to get model information, http StatusCode is %v", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	var filesResp ModeScopeFilesResp
	err = json.Unmarshal(body, &filesResp)
	if err != nil {
		return 0, err
	}

	totalSize := 0
	for _, file := range filesResp.Data.Files {
		totalSize += file.Size
	}

	return int64(math.Ceil(float64(totalSize) / 1024 / 1024 / 1024)), nil
}

func (m *modelscope) ModelGitPath(modelID string) string {
	return fmt.Sprintf("%v%v.git", m.baseURL, modelID)
}

var ModelscopeHub = &modelscope{
	baseURL: "https://www.modelscope.cn/",
	httpClient: &http.Client{
		Timeout: 5 * time.Second,
	},
}
