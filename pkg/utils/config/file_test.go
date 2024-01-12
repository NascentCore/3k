package config

import (
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	got := parse("a: test")
	expected := map[string]string {
		"a": "test",
	}
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("got %v expect %v", got, expected)
	}
}
