package main

import (
	"embed"
)

// Custom log file
const loggerFormat = "csv"
const loggerPath = "/var/log/coraza/coraza-audit-dta.csv"

// Default values for AI verdict and vulnerability type labels.
// These can be modified here without touching the logger logic.
const (
	DefaultAIVerdictLabel           = "malicious" //benign or malicious
	DefaultAIVulnerabilityTypeLabel = "dta"       //none, sqli, xss, dta, etc
)
const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)

// Custom Directives for Coraza
const customCorazaPath = "/etc/coraza/coraza.conf"

const wazuhLoggerFormat = "json"
const wazuhLoggerPath = "/var/log/coraza/coraza-audit.json"

// The annotation below embeds the CRS rules in the "owasp-crs-v4/"" folder
//
//go:embed owasp-crs-v4/*
var crs embed.FS

func GetEmbeddedCRSFS() embed.FS {
	return crs
}

func GetAIMicroserviceURL() string {
	return aiMicroserviceURL
}

func GetTargetAppURL() string {
	return targetAppURL
}
