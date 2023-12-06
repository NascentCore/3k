package storage

import (
	"os"
	"testing"
)

func TestPack(t *testing.T) {
	err := Pack(".", []string{})
	if err != nil {
		t.Error(err)
	}
	os.Remove("./data.zip")
}
