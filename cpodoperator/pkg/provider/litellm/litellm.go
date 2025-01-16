package litellm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

type Model struct {
	ModelName     string        `json:"model_name"`
	LitellmParams LitellmParams `json:"litellm_params"`
	ModelInfo     ModelInfo     `json:"model_info"`
}

type LitellmParams struct {
	APIBase string `json:"api_base"`
	APIKey  string `json:"api_key"`
	Model   string `json:"model"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	DBModel bool   `json:"db_model"`
}

type ModelRemoveRequest struct {
	ID string `json:"id"`
}

type ModelListResponse struct {
	Data []Model `json:"data"`
}

type Litellm struct {
	BaseURL string
	APIKey  string
}

func NewLitellm(baseURL, apiKey string) *Litellm {
	return &Litellm{BaseURL: baseURL, APIKey: apiKey}
}

func (l *Litellm) ListModels() (*ModelListResponse, error) {
	urlStr, err := url.JoinPath(l.BaseURL, "/model/info")
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("GET", urlStr, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+l.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var models ModelListResponse
	err = json.Unmarshal(body, &models)
	if err != nil {
		return nil, err
	}

	return &models, nil
}

func (l *Litellm) AddModel(model Model) error {
	urlStr, err := url.JoinPath(l.BaseURL, "/model/new")
	if err != nil {
		return err
	}

	reqBytes, _ := json.Marshal(model)
	req, err := http.NewRequest("POST", urlStr, bytes.NewBuffer(reqBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+l.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to add model: %s", string(body))
	}

	return nil
}

func (l *Litellm) RemoveModel(model ModelRemoveRequest) error {
	urlStr, err := url.JoinPath(l.BaseURL, "/model/delete")
	if err != nil {
		return err
	}

	reqBytes, _ := json.Marshal(model)
	req, err := http.NewRequest("POST", urlStr, bytes.NewBuffer(reqBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+l.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to remove model: %s", string(body))
	}

	return nil
}
