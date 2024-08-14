package rsa

import (
	"testing"
)

const (
	// Example RSA private key in PEM format
	privateKey = `MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCswmFvmkAbdm7lsXz4/4vHzijkalpvPnY4IdF3Dl7xVv5368AFrRzOYEp9V73YkcSGsV9MoiYC5ktFJncIYnWit66EdC+X6mXxWddQGy+DM3DiiNRMcLg8GyEHvCOo2thc8+JcWb6x+E5IjG8ICZGv6AZXUDM0VjZcoYkdnUPbMVpLkh9j0zVcIWjOLT+IRq1z8NdTXqhCD5wvieGE6i/BvH4ny2uHgYReMwmX98M/FjG0OT0/W9x1mfOWuit/9RXLsf8TQkRSLUpPd05ldRXqa8AD3vpto2NZO1RSDcL5BVq3FXmkEQA0wWliPw5Ta9Po8x6eTlCJrkFZ2T+GhQkRAgMBAAECggEAGY3PF9D7NpS4JqWwVKyIIo/uIItSwS2+EuN9tlyb0psJnPd0OeK3Zhd8LTozTLuTFFFcRVuev8MM0zezV7wopl8BI3qEJ7Z2SdL+NE0xI/kR85BFxuDNf6aLpjdAFxP3+RfiHcLwoCaYfca4DbpMoHZzqaqlSFKQS/dg7s56YQuXYU1eaLVBdvlyDiv9/G4WSNAilCxZc7Hc/KxFfh/r67nb2l1EICeT35015/na7TxhIF9ccurQouAzhSXTLR4dt9DrhyivUGwmzxgKEUvqbffRT41EK2ZGbat/8UVaIIFU24n5BIJTeVgSweUY4MMhLayyCCxQ/2xtbr0TDjh2GQKBgQDkOxZ1CTGdHHrMcIekmyX1cyfZM/nIfhFD0trLypfVxU/IFuzKwaA2JYDXOEcW4j4L0Ku5sTRCQEKYg12KgEdiZ5dw2molUrjtPT9roIwPJxHHSuVkUtNubQPAWePY7t6YE4+GYbtZk65bp29UWPCqDkUNwBkH+u+pUll4Cgy5IwKBgQDBx3ipnNGZnaO7RfvnE3htdKcZykhc/dhnt6cod5PXpSAWBukgIMjZ1gHAJ2T2wHlVWcuaoOyDS1ftISoFv0Lv6490f5y+QyBM7elDwdjsulA4OWidAY3w6P9JRFfhVClCblC/6LkNJ3NjPW3/A00b/sG4pF1/mVKq9aJGDWkKOwKBgE/Aig9poAmrqwmHhQ6zHGeRzunqbK1vyC6wHr6506bipQdhY2tSj9576nLKeqT3eAD+8RMZZg6EkADlXqmIO+maE2RfHlpedrqH/YJpfqfI2kCO3mvZOYLL21S61JC4n9X6d9vYiPQ7U+E5OAD8d1SlWeH9L4IHYPoFCbiVom6LAoGBAI3ZXMScXPpcYQynsDx5DkDhkajZYJth3tYdpCmFTx4ebBxztpekKCL9+44TyF6wiqEl+FsazcdWkeitzDskxPUntH2NJMpKDQ0DYywMbKTtxedCbwfvqV3e0XFqrAHwP7u3UTkKPNwaMudEgo6Ydgu2M0zcVO0g6VjoMn+hNXEdAoGBAIdnFybr9l+uDHdPwHInLbbjNOltG2yphTrc/nM4tw77h0jGHRPfFsO5cSYTE38ZDc68vrE6ZE1qjXDXt7+CAgABnTTt9UQCYiWXNE6NkkyqfupgOCtlLy8a3yG4R+UNF0HmLOzhgoiyDl57iQxAE/RV9DEl+unMHQ1Bc82jF6eh`

	// Corresponding public key in PEM format
	publicKey = `MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArMJhb5pAG3Zu5bF8+P+Lx84o5Gpabz52OCHRdw5e8Vb+d+vABa0czmBKfVe92JHEhrFfTKImAuZLRSZ3CGJ1oreuhHQvl+pl8VnXUBsvgzNw4ojUTHC4PBshB7wjqNrYXPPiXFm+sfhOSIxvCAmRr+gGV1AzNFY2XKGJHZ1D2zFaS5IfY9M1XCFozi0/iEatc/DXU16oQg+cL4nhhOovwbx+J8trh4GEXjMJl/fDPxYxtDk9P1vcdZnzlrorf/UVy7H/E0JEUi1KT3dOZXUV6mvAA976baNjWTtUUg3C+QVatxV5pBEANMFpYj8OU2vT6PMenk5Qia5BWdk/hoUJEQIDAQAB`
)

func TestEncryptDecrypt(t *testing.T) {
	originalText := "Hello, RSA encryption!"

	// Encrypt the original text using the public key
	encryptedData, err := Encrypt(originalText, publicKey)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Decrypt the encrypted data using the private key
	decryptedData, err := Decrypt(encryptedData, privateKey)
	if err != nil {
		t.Fatalf("Decryption failed: %v", err)
	}

	// Check if the decrypted data matches the original text
	if decryptedData != originalText {
		t.Fatalf("Decrypted data does not match original text: got %s, want %s", decryptedData, originalText)
	}

	t.Logf("Encryption and decryption succeeded: %s", decryptedData)
}

func TestDecryptWithInvalidData(t *testing.T) {
	invalidEncryptedData := "invalidbase64data"

	// Attempt to decrypt the invalid data
	_, err := Decrypt(invalidEncryptedData, privateKey)
	if err == nil {
		t.Fatal("Expected an error when decrypting invalid data, but got none")
	}

	t.Logf("Expected error occurred: %v", err)
}

func TestEncryptWithInvalidPublicKey(t *testing.T) {
	invalidPublicKey := "invalidpublickey"

	_, err := Encrypt("Test data", invalidPublicKey)
	if err == nil {
		t.Fatal("Expected an error when encrypting with an invalid public key, but got none")
	}

	t.Logf("Expected error occurred: %v", err)
}
