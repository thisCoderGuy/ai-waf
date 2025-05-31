package main

import (
	"time"
)

// --- Custom Log Entry Structures ---
// These structs define the schema for your JSON log entries.

type CorazaLogEntry struct {
	Timestamp     time.Time            `json:"timestamp"`
	TransactionID string               `json:"transaction_id"`
	ClientIP      string               `json:"client_ip"`
	ClientPort    int                  `json:"client_port"`
	ServerIP      string               `json:"server_ip"`
	ServerPort    int                  `json:"server_port"`
	Request       RequestLogData       `json:"request"`
	Response      ResponseLogData      `json:"response"`
	WAFProcessing WAFProcessingLogData `json:"waf_processing"`
}

type RequestLogData struct {
	Method   string              `json:"method"`
	URI      string              `json:"uri"`
	Protocol string              `json:"protocol"`
	Headers  map[string][]string `json:"headers"`
	Body     string              `json:"body,omitempty"` // Use omitempty for optional body
}

type ResponseLogData struct {
	StatusCode int                 `json:"status_code"`
	Protocol   string              `json:"protocol"`
	Headers    map[string][]string `json:"headers"`
	Body       string              `json:"body,omitempty"`
}

type WAFProcessingLogData struct {
	Interrupted         bool                 `json:"interrupted"`
	InterruptionDetails *InterruptionLogData `json:"interruption_details,omitempty"`
	MatchedRules        []MatchedRuleLogData `json:"matched_rules"`
	AIScore             float64              `json:"ai_score,omitempty"`
	AIVerdict           string               `json:"ai_verdict,omitempty"`
	CorazaDebugLog      string               `json:"coraza_debug_log,omitempty"` // Optional: if you capture debug logs
}

type InterruptionLogData struct {
	RuleID string `json:"rule_id"`
	Status int    `json:"status"`
	Action string `json:"action"`
}

type MatchedRuleLogData struct {
	RuleID      string   `json:"rule_id"`
	Severity    int      `json:"severity"`
	Message     string   `json:"message,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	MatchedData string   `json:"matched_data,omitempty"`
}


// Define a custom type for context keys to avoid collisions.
// This is a best practice for passing values through http.Request contexts.
type contextKey string
// corazaTxContextKey is used to store and retrieve the Coraza transaction from the context.
const corazaTxContextKey contextKey = "coraza-transaction"

const aiScoreContextKey contextKey = "aiScore"
const aiVerdictContextKey contextKey = "aiVerdict"

 const reqBodyContextKey contextKey = "reqBodyBytes"