package common

import "testing"

func TestError(t *testing.T) {
	err := NewError( /*code*/ 1, "error message")
	errStr := err.ToString()
	if errStr != `{"code":1,"msg":"error message","data":null}` {
		t.Errorf("Expected error string, got %s", errStr)
	}
}
