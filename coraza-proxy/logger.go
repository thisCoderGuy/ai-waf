package main

import (
	"log"
	"github.com/corazawaf/coraza/v4/types/variables"
)

// SimpleCorazaLogger implements the Coraza v4 debuglog.Logger interface
// by directly using Go's standard logger.
type SimpleCorazaLogger struct{}

// Debug implements the Debug method of the debuglog.Logger interface.
// Coraza v4 passes a Tx and any additional arguments.
func (l *SimpleCorazaLogger) Debug(tx variables.Transaction, msg string, args ...interface{}) {
	// Format the args as key=value pairs for better readability if they are structured.
	formattedArgs := make([]string, len(args))
	for i, arg := range args {
		formattedArgs[i] = fmt.Sprintf("%v", arg) // Simply format each argument
	}
	log.Printf("[CORAZA_DEBUG] [TX: %s] %s %s", tx.ID(), msg, strings.Join(formattedArgs, " "))
}

// Info implements the Info method.
func (l *SimpleCorazaLogger) Info(tx variables.Transaction, msg string, args ...interface{}) {
	formattedArgs := make([]string, len(args))
	for i, arg := range args {
		formattedArgs[i] = fmt.Sprintf("%v", arg)
	}
	log.Printf("[CORAZA_INFO] [TX: %s] %s %s", tx.ID(), msg, strings.Join(formattedArgs, " "))
}

// Warn implements the Warn method.
func (l *SimpleCorazaLogger) Warn(tx variables.Transaction, msg string, args ...interface{}) {
	formattedArgs := make([]string, len(args))
	for i, arg := range args {
		formattedArgs[i] = fmt.Sprintf("%v", arg)
	}
	log.Printf("[CORAZA_WARN] [TX: %s] %s %s", tx.ID(), msg, strings.Join(formattedArgs, " "))
}

// Error implements the Error method.
func (l *SimpleCorazaLogger) Error(tx variables.Transaction, msg string, args ...interface{}) {
	formattedArgs := make([]string, len(args))
	for i, arg := range args {
		formattedArgs[i] = fmt.Sprintf("%v", arg)
	}
	log.Printf("[CORAZA_ERROR] [TX: %s] %s %s", tx.ID(), msg, strings.Join(formattedArgs, " "))
}

// Fatal implements the Fatal method.
func (l *SimpleCorazaLogger) Fatal(tx variables.Transaction, msg string, args ...interface{}) {
	formattedArgs := make([]string, len(args))
	for i, arg := range args {
		formattedArgs[i] = fmt.Sprintf("%v", arg)
	}
	log.Fatalf("[CORAZA_FATAL] [TX: %s] %s %s", tx.ID(), msg, strings.Join(formattedArgs, " "))
}
////////////////////////////////////////////////////////////
package main

import (
	"bytes"
	"context" // Import context package for proper request context handling
	"encoding/json"
	"fmt"
	"io/ioutil" 
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"

	// Coraza v3, a ModSecurity-compatible WAF engine.
	"github.com/corazawaf/coraza/v3"
	"github.com/corazawaf/coraza/v3/debuglog"
	"github.com/corazawaf/coraza/v3/types" // For Coraza types like Transaction, Interruption, etc.
)

// Define a custom type for context keys to avoid collisions.
// This is a best practice for passing values through http.Request contexts.
type contextKey string

const corazaTxContextKey contextKey = "coraza_tx"

const (
	aiMicroserviceURL = "http://ai-microservice:5000/classify"
	targetAppURL = "http://juice-shop:3000"
)

var waf coraza.WAF
var reverseProxy *httputil.ReverseProxy



func init() {

    myDebugLogger := &SimpleDebugLogger{} // Create an instance of your logger

	cfg := coraza.NewWAFConfig().
		WithRequestBodyAccess().                 // Enable access to the request body for WAF inspection
		WithResponseBodyAccess().                // Enable access to the response body for WAF inspection
		WithDebugLogger(myDebugLogger). // Enable debug logging for Coraza
		WithRootFS(GetEmbeddedCRSFS()) // Call the function from rules.go

	var err error
	waf, err = coraza.NewWAF(cfg)
	if err != nil {
		log.Fatalf("Failed to create WAF: %v", err)
	}

	// Load OWASP CRS rules using Coraza's `Include` directive
	// The paths are relative to the root of the embedded filesystem provided to WithRootFS.
	// In our case, the root is 'owasp-crs-v4' itself.
	crsLoadScript := `
		SecRuleEngine On
		Include owasp-crs-v4/crs-setup.conf.example
		Include owasp-crs-v4/rules/*.conf
	`

	if _, err := waf.NewTransaction().ExecuteMacro(crsLoadScript); err != nil {
		log.Fatalf("Failed to load OWASP CRS rules: %v", err)
	}
	log.Println("Coraza WAF initialized successfully with OWASP CRS v4 embedded rules.")



	// Load our custom Coraza configuration for this proxy.
	// This assumes a file named 'coraza.conf' exists at '/etc/coraza/coraza.conf' in the container.
	corazaConf, err := os.ReadFile("/etc/coraza/coraza.conf")
	if err != nil {
		log.Printf("Failed to read custom Coraza config at /etc/coraza/coraza.conf: %v. Continuing without custom config.", err)
	} else {
		if _, err := waf.NewTransaction().ExecuteMacro(string(corazaConf)); err != nil {
			log.Fatalf("Failed to load custom Coraza config: %v", err)
		}
		log.Println("Custom Coraza config loaded.")
	}

	// 2. Setup Reverse Proxy
	// Parse the target application URL.
	target, err := url.Parse(targetAppURL)
	if err != nil {
		log.Fatalf("Failed to parse target URL: %v", err)
	}
	// Create a new single host reverse proxy that forwards requests to the target.
	reverseProxy = httputil.NewSingleHostReverseProxy(target)

	// Custom Director function to modify requests before forwarding them to the target.
	reverseProxy.Director = func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.Host = target.Host // Important for the target server to receive the correct Host header
		// Remove X-Real-Ip which might be added by previous proxies and can cause issues.
		req.Header.Del("X-Real-Ip")
		// Log the original remote address and the forwarding destination.
		log.Printf("Forwarding request from %s to %s%s", req.RemoteAddr, req.URL.Host, req.URL.Path)
	}

	// ModifyResponse function processes the response from the target application.
	reverseProxy.ModifyResponse = func(res *http.Response) error {
		// Retrieve the Coraza transaction from the request context.
		// This transaction was created and stored in wafHandler.
		tx, ok := res.Request.Context().Value(corazaTxContextKey).(types.Transaction)
		if !ok || tx == nil {
			log.Println("Coraza transaction not found in response context or invalid type. Cannot process response body.")
			return nil
		}

		// Read and process the response body if it exists.
		if res.Body != nil {
			bodyBytes, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return fmt.Errorf("failed to read response body: %w", err)
			}
			res.Body.Close() // Close the original body to prevent resource leaks
			// Restore the body for the client by creating a new io.ReadCloser from the bytes.
			res.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))

			// Process the response body with Coraza.
			tx.ProcessResponseBody(bodyBytes)
		}

		// Check for any interruptions triggered by response body inspection.
		if it := tx.Interruption(); it != nil {
			log.Printf("Response Body Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)
			// Note: Blocking on response body is tricky as the response might have already started
			// being sent to the client. In such cases, you might log, alert, or block
			// subsequent requests from the same source, rather than attempting to block this one.
			// For simplicity, we just log here.
		}

		// Process logging for the entire transaction (request and response phases).
		tx.ProcessLogging()
		return nil
	}

	// ErrorHandler function handles errors that occur during the reverse proxy operation.
	reverseProxy.ErrorHandler = func(rw http.ResponseWriter, req *http.Request, err error) {
		log.Printf("Reverse proxy error for request %s %s: %v", req.Method, req.URL.Path, err)
		http.Error(rw, "Bad Gateway", http.StatusBadGateway)
	}
}

// main function is the entry point of the application.
func main() {
	// Register the wafHandler to handle all incoming HTTP requests.
	http.HandleFunc("/", wafHandler)  
	port := ":8080"
	log.Printf("Coraza Proxy listening on port %s, forwarding to %s", port, targetAppURL)
	// Start the HTTP server. This call blocks until the server stops or an error occurs.
	log.Fatal(http.ListenAndServe(port, nil))
}

// wafHandler processes incoming HTTP requests through Coraza WAF and an AI microservice,
// then forwards them to the target application via a reverse proxy.
func wafHandler(w http.ResponseWriter, r *http.Request) {
	// Create a new Coraza transaction for each incoming request.
	tx := waf.NewTransaction()
	defer tx.Close() // Ensure the transaction is closed to release resources

	// Store the Coraza transaction in the request context.
	// This allows the transaction to be retrieved later in `reverseProxy.ModifyResponse`.
	ctx := context.WithValue(r.Context(), corazaTxContextKey, tx)
	r = r.WithContext(ctx)

	// Log the Coraza transaction ID for debugging and tracing.
	log.Printf("Coraza Transaction ID: %s for %s %s", tx.ID(), r.Method, r.URL.Path)

	// Process request headers with Coraza.
	for k, v := range r.Header {
		tx.AddRequestHeader(k, strings.Join(v, ","))
	}

	// Process request URI and arguments with Coraza.
	tx.ProcessRequestURL(r.URL.String(), r.Method)

	// Process request body if present.
	var bodyBytes []byte
	if r.Body != nil {
		var err error
		// Read the entire request body.
		bodyBytes, err = ioutil.ReadAll(r.Body)
		if err != nil {
			log.Printf("Failed to read request body: %v", err)
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return // Exit if body cannot be read
		}
		// Restore the body for subsequent handlers (AI service and reverse proxy)
		// as `ioutil.ReadAll` consumes the original `r.Body`.
		r.Body = ioutil.NopCloser(bytes.NewBuffer(bodyBytes))
	}
	tx.ProcessRequestBody(bodyBytes)

	// --- AI Microservice Call ---
	// This section integrates with an external AI microservice for additional threat classification.
	aiVerdict := "benign" // Default verdict
	// Only send requests with body, query parameters, or specific methods to AI for efficiency.
	if len(bodyBytes) > 0 || r.URL.RawQuery != "" || r.Method == http.MethodPost || r.Method == http.MethodPut {
		// Prepare the request payload for the AI microservice.
		aiReq := map[string]string{
			"method": r.Method,
			"path":   r.URL.Path,
			"query":  r.URL.RawQuery,
			"body":   string(bodyBytes),
			// Additional request details (headers, client IP) could be added here.
		}
		aiReqJSON, _ := json.Marshal(aiReq) // Marshal to JSON, ignoring error for simplicity in example

		log.Printf("Sending request to AI microservice: %s", aiMicroserviceURL)
		// Make an HTTP POST request to the AI microservice.
		aiResp, err := http.Post(aiMicroserviceURL, "application/json", bytes.NewBuffer(aiReqJSON))
		if err != nil {
			log.Printf("Error calling AI microservice: %v", err)
		} else {
			defer aiResp.Body.Close() // Ensure response body is closed
			if aiResp.StatusCode == http.StatusOK {
				aiRespBody, _ := ioutil.ReadAll(aiResp.Body) // Read AI response, ignoring error
				var aiRespMap map[string]interface{}
				// Unmarshal AI response JSON.
				if err := json.Unmarshal(aiRespBody, &aiRespMap); err == nil {
					if verdict, ok := aiRespMap["verdict"].(string); ok {
						aiVerdict = verdict
						log.Printf("AI Microservice verdict: %s", aiVerdict)
					}
					if score, ok := aiRespMap["score"].(float64); ok {
						log.Printf("AI Microservice score: %.2f", score)
						// Inject AI score into Coraza transaction for rule evaluation.
						// A custom Coraza rule could then use TX:AI_SCORE, e.g.:
						// SecRule TX:AI_SCORE "@gt 0.8" "id:1000,phase:2,deny,msg:'AI detected high score'"
						tx.Collection(types.TxCollection).Set("AI_SCORE", fmt.Sprintf("%.2f", score))
					}
				}
			} else {
				log.Printf("AI Microservice returned non-OK status: %d", aiResp.StatusCode)
			}
		}
	}

	// --- Coraza Interruption and AI Decision Enforcement ---
	// Process the request with Coraza rules and check for any immediate interruption.
	it := tx.ProcessRequest()

	// If Coraza didn't block the request, but the AI microservice deemed it malicious,
	// we can force an interruption based on the AI's verdict.
	if it == nil && aiVerdict == "malicious" {
		log.Println("AI detected malicious request, forcing Coraza interruption.")
		// Manually create an interruption struct to simulate a WAF block.
		it = &types.Interruption{
			RuleID: "AI_DECISION",                        // A custom identifier for blocks initiated by AI
			Status: 403,                                  // HTTP status code (e.g., Forbidden)
			Action: "deny",                               // The action taken
			Msg:    "Blocked by AI microservice verdict", // Optional message
		}
	}

	// Check for interruption again (either from Coraza rules or AI-forced).
	// If `it` is still nil, it means no interruption occurred.
	if it != nil {
		log.Printf("Coraza WAF Blocked Request! Action: %s, Rule ID: %s, Status: %d, Message: %s", it.Action, it.RuleID, it.Status, it.Msg)
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(it.Status)
		// Write a response to the client indicating the block.
		_, _ = w.Write([]byte(fmt.Sprintf("Coraza WAF Blocked: %s (Rule ID: %s, Message: %s)", it.Action, it.RuleID, it.Msg)))
		tx.ProcessLogging() // Ensure logging is processed for blocked requests.
		return              // Stop processing and return the response.
	}

	// If no interruption occurred, forward the request to the target application.
	reverseProxy.ServeHTTP(w, r)

	// Note: Response body inspection and final `tx.ProcessLogging()` for the full transaction
	// are handled within the `reverseProxy.ModifyResponse` function.
}
