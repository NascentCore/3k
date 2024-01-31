package sxwl

import (
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/NascentCore/cpodoperator/pkg/resource"
)

func Test_sxwl_GetAssignedJobList(t *testing.T) {
	baseURL, _ := os.LookupEnv("SXWL_BASE_URL")
	accessKey, _ := os.LookupEnv("SXWL_ACCESS_KEY")
	identity, _ := os.LookupEnv("SXWL_IDENTITY")

	if baseURL == "" {
		t.Skip("skip test")
	}

	s := &sxwl{
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		baseURL:   baseURL,
		accessKey: accessKey,
		identity:  identity,
	}

	tests := []struct {
		name    string
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name:    "ok",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tjobs, ijobs, err := s.GetAssignedJobList()
			fmt.Println(tjobs)
			fmt.Println(ijobs)
			if (err != nil) != tt.wantErr {
				t.Errorf("sxwl.GetAssignedJobList() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func Test_sxwl_HeartBeat(t *testing.T) {
	baseURL, _ := os.LookupEnv("SXWL_BASE_URL")
	accessKey, _ := os.LookupEnv("SXWL_ACCESS_KEY")
	identity, _ := os.LookupEnv("SXWL_IDENTITY")

	if baseURL == "" {
		t.Skip("skip test")
	}

	s := &sxwl{
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		baseURL:   baseURL,
		accessKey: accessKey,
		identity:  identity,
	}

	tests := []struct {
		name    string
		wantErr bool
	}{
		// TODO: Add test cases.
		{
			name:    "ok",
			wantErr: false,
		},
	}
	var testPayload HeartBeatPayload
	testPayload = HeartBeatPayload{
		CPodID:    "",
		JobStatus: []State{},
		ResourceInfo: resource.CPodResourceInfo{
			CPodID:      "cpod0001",
			CPodVersion: "1.0",
			GPUSummaries: []resource.GPUSummary{{
				Vendor:      "nvidia",
				Prod:        "3090",
				Total:       1,
				Allocatable: 1,
			}},
			Caches: []resource.Cache{},
			Nodes:  []resource.NodeInfo{},
		},
		UpdateTime: time.Time{},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := s.HeartBeat(testPayload)
			if (err != nil) != tt.wantErr {
				t.Errorf("sxwl.HeartBeat() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}
