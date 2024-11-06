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

func TestParseBytes(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int64
		wantErr bool
	}{
		{
			name:    "正确的GB格式",
			input:   "5.2 GB",
			want:    5583457484, // 5.2 * 1024 * 1024 * 1024
			wantErr: false,
		},
		{
			name:    "正确的MB格式",
			input:   "3MB",
			want:    3145728, // 3 * 1024 * 1024
			wantErr: false,
		},
		{
			name:    "正确的KB格式",
			input:   "1024.5KB",
			want:    1049088, // 1024.5 * 1024
			wantErr: false,
		},
		{
			name:    "带额外空格的格式",
			input:   "  2.5  GB  ",
			want:    2684354560, // 2.5 * 1024 * 1024 * 1024
			wantErr: false,
		},
		{
			name:    "错误的单位",
			input:   "5.2 TB",
			want:    0,
			wantErr: true,
		},
		{
			name:    "错误的数字格式",
			input:   "abc GB",
			want:    0,
			wantErr: true,
		},
		{
			name:    "空字符串",
			input:   "",
			want:    0,
			wantErr: true,
		},
		{
			name:    "无单位",
			input:   "1024",
			want:    0,
			wantErr: true,
		},
		{
			name:    "错误的格式",
			input:   "1024Bytes",
			want:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseBytes(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseBytes() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ParseBytes() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestParseBytesAndFormatBytes 测试 ParseBytes 和 FormatBytes 的互操作性
func TestParseBytesAndFormatBytes(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "2.50 GB",
			input: "2.50 GB",
		},
		{
			name:  "1024.00 MB",
			input: "1024.00 MB",
		},
		{
			name:  "2048.00 KB",
			input: "2048.00 KB",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 先解析字符串到字节数
			bytes, err := ParseBytes(tt.input)
			if err != nil {
				t.Fatalf("ParseBytes() error = %v", err)
			}

			// 再将字节数转回字符串
			got := FormatBytes(bytes)

			// 比较结果（注意：由于浮点数精度的原因，我们需要再次解析结果进行比较）
			gotBytes, err := ParseBytes(got)
			if err != nil {
				t.Fatalf("ParseBytes() error = %v", err)
			}

			// 允许 1% 的误差
			diff := float64(gotBytes-bytes) / float64(bytes)
			if diff < -0.01 || diff > 0.01 {
				t.Errorf("转换来回不匹配。原始值：%v，得到：%v，误差：%.2f%%", tt.input, got, diff*100)
			}
		})
	}
}
