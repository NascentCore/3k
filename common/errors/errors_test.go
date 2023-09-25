package errors

import (
	"testing"
)

// TestUnImpl tests UnImpl returns an error with expected error message.
func TestUnImpl(t *testing.T) {
	if UnImpl("test").Error() != "test is not implemented yet" {
		t.Fatal(UnImpl("test"))
	}
}
