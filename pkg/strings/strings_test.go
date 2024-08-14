package strings

import (
	"testing"
	"unicode"
)

func TestRandomString(t *testing.T) {
	type args struct {
		length           int
		includeUppercase bool
		includeNumbers   bool
		includeSpecial   bool
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "All character types",
			args: args{
				length:           12,
				includeUppercase: true,
				includeNumbers:   true,
				includeSpecial:   true,
			},
			wantErr: false,
		},
		{
			name: "Lowercase only",
			args: args{
				length:           10,
				includeUppercase: false,
				includeNumbers:   false,
				includeSpecial:   false,
			},
			wantErr: false,
		},
		{
			name: "Include uppercase and numbers",
			args: args{
				length:           8,
				includeUppercase: true,
				includeNumbers:   true,
				includeSpecial:   false,
			},
			wantErr: false,
		},
		{
			name: "Invalid length",
			args: args{
				length:           0,
				includeUppercase: true,
				includeNumbers:   true,
				includeSpecial:   true,
			},
			wantErr: true,
		},
		{
			name: "No uppercase, numbers, or special characters",
			args: args{
				length:           12,
				includeUppercase: false,
				includeNumbers:   false,
				includeSpecial:   false,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RandomString(tt.args.length, tt.args.includeUppercase, tt.args.includeNumbers, tt.args.includeSpecial)
			if (err != nil) != tt.wantErr {
				t.Errorf("RandomString() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(got) != tt.args.length {
					t.Errorf("RandomString() length = %v, want %v", len(got), tt.args.length)
				}
				hasUppercase := false
				hasNumbers := false
				hasSpecial := false

				for _, c := range got {
					if unicode.IsUpper(c) {
						hasUppercase = true
					}
					if unicode.IsDigit(c) {
						hasNumbers = true
					}
					if unicode.IsPunct(c) || unicode.IsSymbol(c) {
						hasSpecial = true
					}
				}

				if tt.args.includeUppercase && !hasUppercase {
					t.Errorf("RandomString() missing uppercase character")
				}
				if tt.args.includeNumbers && !hasNumbers {
					t.Errorf("RandomString() missing numeric character")
				}
				if tt.args.includeSpecial && !hasSpecial {
					t.Errorf("RandomString() missing special character")
				}
			}
		})
	}
}
