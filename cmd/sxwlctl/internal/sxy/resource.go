package sxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/spf13/viper"
)

type Resource struct {
	ResourceID   string `json:"resource_id"`
	ResourceType string `json:"resource_type"`
	ResourceName string `json:"resource_name"`
	ResourceSize int64  `json:"resource_size"`
	IsPublic     bool   `json:"is_public"`
	UserID       string `json:"user_id"`
	Meta         string `json:"meta"`
}

func AddResource(token string, resource Resource) error {
	// Convert resource to JSON
	resourceJSON, err := json.Marshal(resource)
	if err != nil {
		return fmt.Errorf("Error marshaling resource to JSON: %s\n", err)
	}

	// Create a new request using http.NewRequest
	req, err := http.NewRequest("POST", viper.GetString("resource_url"), bytes.NewBuffer(resourceJSON))
	if err != nil {
		return fmt.Errorf("Error creating request: %s\n", err)
	}

	// Add an 'Authorization' header to the request
	req.Header.Add("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	// Send the request using http.Client
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Error sending request to API endpoint: %s\n", err)
	}
	defer resp.Body.Close()

	// Check if the response status code indicates success (200 OK)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("request failed with status code: %d", resp.StatusCode)
	}

	return nil
}
