package errors

import (
	"testing"
)

func TestUnImpl(t *testing.T) {
	if UnImpl("test").Error() != "test is not implemented yet" {
		t.Fatal(UnImpl("test"))
	}
}
