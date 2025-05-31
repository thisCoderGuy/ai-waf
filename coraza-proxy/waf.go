package main

import (
	"fmt"
	//"log"
	"github.com/corazawaf/coraza/v3"
	"io/ioutil"
	"os"
)



var waf coraza.WAF

func InitializeWAF() error {
	customDirectives, err := loadCustomDirectives(customCorazaPath)
	if err != nil {
		return fmt.Errorf("failed to load custom Coraza configuration: %w", err)
	}

	cfg := coraza.NewWAFConfig().
		WithRequestBodyAccess().  // Enable access to the request body for WAF inspection
		WithResponseBodyAccess(). // Enable access to the response body for WAF inspection
		WithRootFS(GetEmbeddedCRSFS()). // Call the function from rules.go
		WithDirectives(customDirectives)

	waf, err = coraza.NewWAF(cfg)
	if err != nil {
		return fmt.Errorf("failed to create WAF: %w", err)
	}

	//log.Println("Coraza WAF initialized successfully with OWASP CRS v4 embedded rules!")
	return nil
}

func GetWAF() coraza.WAF {
	return waf
}

func loadCustomDirectives(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	content, err := ioutil.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("failed to read config file: %w", err)
	}

	return string(content), nil
}