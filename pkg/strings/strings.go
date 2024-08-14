package strings

import (
	"crypto/rand"
	"errors"
	"math/big"
)

// RandomString generates a random password with the specified character complexity.
func RandomString(length int, includeUppercase, includeNumbers, includeSpecial bool) (string, error) {
	if length <= 0 {
		return "", errors.New("length must be greater than 0")
	}

	lowercase := "abcdefghijklmnopqrstuvwxyz"
	uppercase := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	numbers := "0123456789"
	special := "!@#$%^&*"

	// The default character pool includes lowercase letters.
	characterPool := lowercase

	if includeUppercase {
		characterPool += uppercase
	}
	if includeNumbers {
		characterPool += numbers
	}
	if includeSpecial {
		characterPool += special
	}

	if characterPool == "" {
		return "", errors.New("at least one character type must be selected")
	}

	// Ensure that the generated password includes at least one of each selected character type
	var password []byte
	if includeUppercase {
		randomChar, err := getRandomChar(uppercase)
		if err != nil {
			return "", err
		}
		password = append(password, randomChar)
	}
	if includeNumbers {
		randomChar, err := getRandomChar(numbers)
		if err != nil {
			return "", err
		}
		password = append(password, randomChar)
	}
	if includeSpecial {
		randomChar, err := getRandomChar(special)
		if err != nil {
			return "", err
		}
		password = append(password, randomChar)
	}

	// Fill the rest of the password with random characters from the full pool
	for len(password) < length {
		randomChar, err := getRandomChar(characterPool)
		if err != nil {
			return "", err
		}
		password = append(password, randomChar)
	}

	// Shuffle the password to avoid predictable patterns
	shuffledPassword, err := shuffle(password)
	if err != nil {
		return "", err
	}

	return string(shuffledPassword), nil
}

// getRandomChar returns a random character from a given string.
func getRandomChar(pool string) (byte, error) {
	poolSize := big.NewInt(int64(len(pool)))
	n, err := rand.Int(rand.Reader, poolSize)
	if err != nil {
		return 0, err
	}
	return pool[n.Int64()], nil
}

// shuffle randomly shuffles a byte slice.
func shuffle(data []byte) ([]byte, error) {
	for i := range data {
		j, err := getRandomIndex(len(data))
		if err != nil {
			return nil, err
		}
		data[i], data[j] = data[j], data[i]
	}
	return data, nil
}

// getRandomIndex returns a random index within a range.
func getRandomIndex(length int) (int, error) {
	lengthSize := big.NewInt(int64(length))
	n, err := rand.Int(rand.Reader, lengthSize)
	if err != nil {
		return 0, err
	}
	return int(n.Int64()), nil
}
