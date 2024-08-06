package sxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sxwl/3k/internal/scheduler/types"

	"github.com/spf13/viper"
)

func GetAccessByToken(token string) (id, key string, userID string, isAdmin bool, err error) {
	// Create a new request using http.NewRequest
	req, err := http.NewRequest("GET", viper.GetString("auth_url"), nil)
	if err != nil {
		err = fmt.Errorf("Error creating request: %s\n", err)
		return
	}

	// Add an 'Authorization' header to the request
	req.Header.Add("Authorization", "Bearer "+token)

	// Send the request using http.Client
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		err = fmt.Errorf("Error sending request to API endpoint: %s\n", err)
		return
	}
	defer resp.Body.Close()

	// Check if the response status code indicates success (200 OK)
	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("auth_url request failed with status code: %d", resp.StatusCode)
		return
	}

	// Decode the JSON response into the UploaderAccessResp struct
	var response types.UploaderAccessResp
	if err = json.NewDecoder(resp.Body).Decode(&response); err != nil {
		err = fmt.Errorf("Error decoding JSON response: %s\n", err)
		return
	}

	return response.AccessID, response.AccessKey, response.UserID, response.IsAdmin, err
}
