package rsa

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
)

func Decrypt(encryptedData, privateKey string) (string, error) {
	// Your RSA private key in PEM format
	privateKeyPEM := fmt.Sprintf(`-----BEGIN RSA PRIVATE KEY-----
%s
-----END RSA PRIVATE KEY-----`, privateKey)

	// Decode the base64 encoded data
	decodedData, err := base64.StdEncoding.DecodeString(encryptedData)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64 data err=%s", err)
	}

	// Parse the PEM encoded private key
	block, _ := pem.Decode([]byte(privateKeyPEM))
	if block == nil {
		return "", fmt.Errorf("failed to parse PEM block containing the key")
	}

	// Parse the PKCS#8 private key
	privateKeyInterface, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return "", fmt.Errorf("failed to parse PKCS#8 private key err=%s", err)
	}

	privKey, ok := privateKeyInterface.(*rsa.PrivateKey)
	if !ok {
		return "", fmt.Errorf("not an RSA private key")
	}

	// Decrypt the data
	decryptedData, err := rsa.DecryptPKCS1v15(rand.Reader, privKey, decodedData)
	if err != nil {
		return "", fmt.Errorf("failed to decrypt data err=%s", err)
	}

	return string(decryptedData), nil
}

func Encrypt(data, publicKey string) (string, error) {
	// Your RSA public key in PEM format
	publicKeyPEM := fmt.Sprintf(`-----BEGIN PUBLIC KEY-----
%s
-----END PUBLIC KEY-----`, publicKey)

	// Parse the PEM encoded public key
	block, _ := pem.Decode([]byte(publicKeyPEM))
	if block == nil {
		return "", fmt.Errorf("failed to parse PEM block containing the public key")
	}

	// Parse the PKCS#1 or PKCS#8 public key
	pubKeyInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return "", fmt.Errorf("failed to parse public key err=%s", err)
	}

	pubKey, ok := pubKeyInterface.(*rsa.PublicKey)
	if !ok {
		return "", fmt.Errorf("not an RSA public key")
	}

	// Encrypt the data
	encryptedData, err := rsa.EncryptPKCS1v15(rand.Reader, pubKey, []byte(data))
	if err != nil {
		return "", fmt.Errorf("failed to encrypt data err=%s", err)
	}

	// Encode the encrypted data as base64
	encodedData := base64.StdEncoding.EncodeToString(encryptedData)

	return encodedData, nil
}
