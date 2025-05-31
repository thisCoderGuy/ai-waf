package main

import (
	"bytes"
	"context"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
	
	"github.com/corazawaf/coraza/v3/types"
)

// WAFHandler processes incoming HTTP requests through Coraza WAF and an AI microservice,
// then forwards them to the target application via a reverse proxy, if not intercepted.

// Client Request
//
//	↓
//
// [wafHandler]
//
//		↳ Coraza WAF: analyze  request headers, request body, URI (Phases 1 & 2)
//		↳ AI Microservice: classify threat level
//		↳ If malicious → block, enrich context with ai_malicious code, then  log (to be sent to wazuh)
//		↳ If safe → pass to reverseProxy
//	    [reverseProxy]
//		    ↳ Forward to backend (e.g., Juice Shop)
//		    ↳ Get response
//		    ↳ Coraza WAF: analyze response (Phase 3 & 4)
//		    ↳ Return response to client, if not intercepted.
func WAFHandler(w http.ResponseWriter, r *http.Request) {
	//log.Println(">>> WAFHandler called for "+ r.URL.Path)
	//GetGlobalLogger().LogInfo(">>> WAFHandler called for "+ r.URL.Path)

	// Create a new Coraza transaction for each incoming request.
	tx := GetWAF().NewTransaction()
	defer tx.Close() // Ensure the transaction is closed to release resources

	ctx := r.Context()

	// Store the Coraza transaction in the request context.
	// This allows the transaction to be retrieved later in `reverseProxy.ModifyResponse`.
	ctx = context.WithValue(ctx, corazaTxContextKey, tx)

	

	// Log the Coraza transaction ID for debugging and tracing.
	//log.Printf("Coraza Transaction ID: %s for %s %s", tx.ID(), r.Method, r.URL.Path)
	//GetGlobalLogger().LogInfo("Coraza Transaction for "+ r.URL.Path)


	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0. Coraza Transaction Initial Setup. Does not trigger any rule.
	// Phase 0.0 ProcessConnection() is used to log metadata about the TCP connection that initiated the request (Client IP and port, Server IP and port).
	// Parse client IP and port
	clientIP, clientPortStr, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		http.Error(w, "Invalid remote address", http.StatusInternalServerError)
		return
	}
	clientPort, _ := strconv.Atoi(clientPortStr)

	// Parse server IP and port
	serverHost := r.Host
	if _, _, err := net.SplitHostPort(serverHost); err != nil {
		// Append default port if not specified
		serverHost += ":80"
	}
	serverIP, serverPortStr, err := net.SplitHostPort(serverHost)
	if err != nil {
		http.Error(w, "Invalid host address", http.StatusInternalServerError)
		return
	}
	serverPort, _ := strconv.Atoi(serverPortStr)

	// Now call ProcessConnection
	tx.ProcessConnection(clientIP, clientPort, serverIP, serverPort)
	if it := tx.Interruption(); it != nil {
		//log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "WAF headers check failed", http.StatusBadRequest)
			tx.ProcessLogging()
			if logger := GetGlobalLogger(); logger != nil {
				logger.LogTransaction(tx, r, nil, -1, "not evaluated")
			}
			return
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0.1 ProcessURI() is used to used to initialize the transaction with key elements of the HTTP request line
	// It sets the following: Request URI (e.g., /login?user=admin), HTTP method (e.g., GET, POST), HTTP version (e.g., HTTP/1.1, HTTP/2)
	// Process request URI and arguments with Coraza.
	uri := r.URL.String()
	if r.URL.RawQuery != "" {
		uri = uri + "?" + r.URL.RawQuery
	}
	tx.ProcessURI(uri, r.Method, r.Proto)

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 0.2 AddRequestHeader() is used to inject or append HTTP request headers into the transaction object
	// Process request headers with Coraza.
	for k, v := range r.Header {
		tx.AddRequestHeader(k, strings.Join(v, ","))
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 1. ProcessRequestHeaders() evaluates the HTTP request headers against ModSecurity/OWASP rules.
	it := tx.ProcessRequestHeaders()

	// Check for any interruptions triggered by request inspection.
	if it := tx.Interruption(); it != nil {
		//log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "WAF headers check failed", http.StatusBadRequest)
			tx.ProcessLogging()
			if logger := GetGlobalLogger(); logger != nil {
				logger.LogTransaction(tx, r, nil, -1, "not evaluated")
			}
			
			return
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Phase 2. ProcessRequestBody() inspects the HTTP request body (e.g., POST data, JSON payloads, form submissions)  for malicious content based on the loaded WAF rules.
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

	// If you read the request body in wafHandler and want to log it in ModifyResponse:
    ctx = context.WithValue(ctx, reqBodyContextKey, bodyBytes)

	tx.ProcessRequestBody()

	// Check for any interruptions triggered by request inspection.
	if it := tx.Interruption(); it != nil {
		//log.Printf("Request Coraza Interruption detected! Action: %s, Rule ID: %s, Status: %d", it.Action, it.RuleID, it.Status)

		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "Request blocked by WAF", http.StatusForbidden)
			tx.ProcessLogging()
			if logger := GetGlobalLogger(); logger != nil {
				logger.LogTransaction(tx, r, nil, -1, "not evaluated")
			}			
			return
		}

	}

	aiVerdict, aiScore := CallAIMicroservice(r, bodyBytes, tx)
    
    // Cache the AI score and verdict in the context
        ctx = context.WithValue(ctx, aiScoreContextKey, aiScore) // Store float64
        ctx = context.WithValue(ctx, aiVerdictContextKey, aiVerdict) // Store string
    
    
    

    // Update the request with the new context 
    r = r.WithContext(ctx)


	// ---  AI Decision Enforcement ---

	// If Coraza didn't block the request, but the AI microservice deemed it malicious,
	// we can force an interruption based on the AI's verdict.
	if aiVerdict == "malicious" {
		log.Println("AI detected malicious request, forcing Coraza interruption.")
		// Manually create an interruption struct to simulate a WAF block.
		it = &types.Interruption{
			RuleID: 666,    // A custom identifier for blocks initiated by AI
			Status: 403,    // HTTP status code (e.g., Forbidden)
			Action: "deny", // The action taken
		}
	}

	// Check for interruption again (either from Coraza rules or AI-forced).
	// If `it` is still nil, it means no interruption occurred.
	if it != nil {
		//log.Printf("Coraza WAF Blocked Request! Action: %s, Rule ID: %d, Status: %d", it.Action, it.RuleID, it.Status)
		
		// Handle the interruption, if any
		if it.Action == "deny" {
			http.Error(w, "Request blocked by WAF", http.StatusForbidden)
			tx.ProcessLogging()
			if logger := GetGlobalLogger(); logger != nil {
				logger.LogTransaction(tx, r, nil, aiScore, aiVerdict)
			}
			return
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// If no interruption occurred, forward the request to the target application by using the reverse proxy.
	reverseProxy.ServeHTTP(w, r)
	// Note: Phase 3 (Response Headers Inspection), Phase 4 (Response body inspection) and Phase 5 (`tx.ProcessLogging()`) for the full transaction
	// are handled within the `reverseProxy.ModifyResponse` function.
}