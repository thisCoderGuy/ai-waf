package main

import (
	"time"
)

// --- Custom Log Entry Structures ---
// These structs define the schema for your JSON log entries.
// CorazaLogEntry represents the structure of a log entry for JSON output.
type CorazaLogEntry struct {
	Timestamp     time.Time `json:"timestamp"`
	TransactionID string    `json:"transaction_id"`
	ClientIP      string    `json:"client_ip"`
	ClientPort    int       `json:"client_port"`
	ServerIP      string    `json:"server_ip"`
	ServerPort    int       `json:"server_port"`
	Request       RequestLogData `json:"request"`
	Response      *ResponseLogData `json:"response,omitempty"`
	WAFProcessing WAFLogData `json:"waf_processing"`
	Calculated    CalculatedFields `json:"calculated_fields"`
}

// RequestLogData contains details about the HTTP request.
type RequestLogData struct {
	Method          string          `json:"method"`
	URI             string          `json:"uri"`
	Path            string          `json:"path"`
	Query           string          `json:"query"`
	Protocol        string          `json:"protocol"`
	Headers         ParsedHeaders   `json:"headers"` // Parsed headers
	Body            string          `json:"body"`
}

// ParsedHeaders separates common headers into individual fields.
type ParsedHeaders struct {
	UserAgent      string              `json:"user_agent,omitempty"`
	Referer        string              `json:"referer,omitempty"`
	AcceptEncoding string              `json:"accept_encoding,omitempty"`
	ContentType    string              `json:"content_type,omitempty"`
	Accept         string              `json:"accept,omitempty"`
	Cookie         string              `json:"cookie,omitempty"`
	Connection     string              `json:"connection,omitempty"`
	OtherHeaders   map[string][]string `json:"other_headers,omitempty"` // For headers not explicitly separated
}

// ResponseLogData contains details about the HTTP response.
type ResponseLogData struct {
	StatusCode int               `json:"status_code"`
	Protocol   string            `json:"protocol"`
	Headers    map[string][]string `json:"headers"`
	Body       string            `json:"body"`
}

// WAFLogData contains details about WAF processing.
type WAFLogData struct {
	Interrupted         bool                 `json:"interrupted"`
	InterruptionDetails *InterruptionLogData `json:"interruption_details,omitempty"`
	MatchedRules        []MatchedRuleLogData `json:"matched_rules,omitempty"`
	AIScore             float64              `json:"ai_score"`
	AIVerdict           string               `json:"ai_verdict"`
}

// InterruptionLogData contains details if the transaction was interrupted.
type InterruptionLogData struct {
	RuleID string `json:"rule_id"`
	Status int    `json:"status"`
	Action string `json:"action"`
}

// MatchedRuleLogData contains details about a matched WAF rule.
type MatchedRuleLogData struct {
	RuleID  string   `json:"rule_id"`
	Message string   `json:"message"`
	Tags    []string `json:"tags"`
}

// CalculatedFields holds all the newly calculated metrics.
type CalculatedFields struct {
	RequestLength                 int     `json:"request_length"`
	PathLength                    int     `json:"path_length"`
	QueryLength                   int     `json:"query_length"`
	BodyLength                    int     `json:"body_length"`
	NumParams                     int     `json:"num_params"`
	LongestParamValueLength       int     `json:"longest_param_value_length"`
	NumSpecialChars               int     `json:"num_special_chars"`
	NumEncodedChars               int     `json:"num_encoded_chars"`
	NumDoubleEncodedChars         int     `json:"num_double_encoded_chars"`
	RatioSpecialCharsToTotalChars float64 `json:"ratio_special_chars_to_total_chars"`
	NumKeywords                   int     `json:"num_keywords"`
	NumUniqueChars                int     `json:"num_unique_chars"`
	IsJSONBody                    bool    `json:"is_json_body"`
	IsXMLBody                     bool    `json:"is_xml_body"`
	IsMultipartForm               bool    `json:"is_multipart_form"`
	HasNumericPathSegment         bool    `json:"has_numeric_path_segment"`
	DepthOfPath                   int     `json:"depth_of_path"`
	UsesBase64Encoding            bool    `json:"uses_base64_encoding"`
	UsesHexEncoding               bool    `json:"uses_hex_encoding"`
	HasMultipleContentLength      bool    `json:"has_multiple_content_length"`
	InvalidHTTPVersion            bool    `json:"invalid_http_version"`
	NumHTTPHeaders                int     `json:"num_http_headers"`
	TimeOfDayHour                 int     `json:"time_of_day_hour"`
	TimeOfDayDayOfWeek            string  `json:"time_of_day_day_of_week"`
}

/////////////////////



// Define a custom type for context keys to avoid collisions.
// This is a best practice for passing values through http.Request contexts.
type contextKey string
// corazaTxContextKey is used to store and retrieve the Coraza transaction from the context.
const corazaTxContextKey contextKey = "coraza-transaction"

const aiScoreContextKey contextKey = "aiScore"
const aiVerdictContextKey contextKey = "aiVerdict"

 const reqBodyContextKey contextKey = "reqBodyBytes"