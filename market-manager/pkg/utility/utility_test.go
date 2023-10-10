package utility

import "testing"

func TestGetJobInfo(t *testing.T) {
	_, err := GetJobInfo("test")
	if err == nil {
		t.Errorf("Expected failure, got nil error")
	}
}
