package logic

import "testing"

func TestSelectCpods(t *testing.T) {
	_, err := SelectCpods(nil)
	if err.Error() != "cpods is empty" {
		t.Errorf("Expected no error, got %v", err)
	}
}
