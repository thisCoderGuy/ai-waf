package main

import (
	"embed"
)

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)



//Custom Directives for Coraza
const customCorazaPath = "/etc/coraza/coraza.conf"

//Custom log file
// The path /var/log/coraza/coraza-audit.log needs write permissions for the user running the app.
// Consider making this configurable via env var or flag.
const loggerFormat = "csv"
const loggerPath ="/var/log/coraza/coraza-audit-benign.csv"
// Default values for AI verdict and vulnerability type labels.
// These can be modified here without touching the logger logic.
const (
	DefaultAIVerdictLabel         = "benign" //benign or malicious
	DefaultAIVulnerabilityTypeLabel = "benign"    //none, sqli, xss, etc
)

const wazuhLoggerFormat = "json"
const wazuhLoggerPath ="/var/log/coraza/coraza-audit.json"



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

