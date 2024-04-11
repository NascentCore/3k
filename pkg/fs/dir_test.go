package fs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFormatBytes(t *testing.T) {
	type args struct {
		bytes int64
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{
			name: "Test for One KB",
			args: args{bytes: 1024},
			want: "1.00 KB",
		},
		{
			name: "Test for KB",
			args: args{bytes: 1500},
			want: "1.46 KB",
		},
		{
			name: "Test for One MB",
			args: args{bytes: 1048576},
			want: "1.00 MB",
		},
		{
			name: "Test for MB",
			args: args{bytes: 1572864},
			want: "1.50 MB",
		},
		{
			name: "Test for One GB",
			args: args{bytes: 1073741824},
			want: "1.00 GB",
		},
		{
			name: "Test for GB",
			args: args{bytes: 1610612736},
			want: "1.50 GB",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FormatBytes(tt.args.bytes); got != tt.want {
				t.Errorf("FormatBytes() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestListFilesInDir tests the ListFilesInDir function.
func TestListFilesInDir(t *testing.T) {
	// Setup a temporary directory.
	tmpDir := t.TempDir()

	// Create some test files.
	fileNames := []string{"test1.txt", "test2.txt", "example.go", "readme.md"}
	for _, fName := range fileNames {
		tmpFile, err := os.Create(filepath.Join(tmpDir, fName))
		if err != nil {
			t.Fatal(err)
		}
		tmpFile.Close()
	}

	// Define test cases
	tests := []struct {
		ext      string
		expected int
	}{
		{".txt", 2},  // Expect 2 .txt files
		{".go", 1},   // Expect 1 .go file
		{"", 4},      // Expect all 4 files
		{".json", 0}, // Expect no .json files
	}

	// Execute test cases
	for _, tc := range tests {
		result, err := ListFilesInDir(tmpDir, tc.ext)
		if err != nil {
			t.Errorf("Failed to list files: %v", err)
			continue
		}
		if len(result) != tc.expected {
			t.Errorf("Expected %d files for extension '%s', got %d", tc.expected, tc.ext, len(result))
		}
	}
}
