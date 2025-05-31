package main

import (
	"embed"
)

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL      = "http://juice-shop:3000"
)

//Custom log file
// The path /var/log/coraza/coraza-audit.log needs write permissions for the user running the app.
// Consider making this configurable via env var or flag.
const	loggerPath = "/var/log/coraza/coraza-audit.csv"

//Custom Directives for Craza
const customCorazaPath = "/etc/coraza/coraza.conf"

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

