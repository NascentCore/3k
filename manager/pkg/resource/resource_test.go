package resource

import (
	"fmt"
	"testing"
)

func TestGetResourceDesc(t *testing.T) {
	// TODO:
}

func TestToJson(t *testing.T) {
	var tmp CPodResourceDesc
	tmp.Nodes = []Node{Node{GPUs: GPUs{Individuals: []GPU{GPU{}}}}}
	r := tmp.ToJson()
	fmt.Println(string(r))
	if r == nil {
		t.Error("marshal CPodResourceDesc failed")
	}
}
