package main

import (
	"log"
	"net/http"
	"time" // Added for potential future use, e.g. server timeouts
)


func init() {
	var err error

	loggerConfig := LoggerConfig{
		Format:   loggerFormat,
		Filename: loggerPath,
	}
	// Create logger that saves to file
	loggerInstance, err := NewCorazaLogger(loggerConfig)
	if err != nil {
		log.Fatalf("Failed to create Coraza logger: %v", err)
	}
	SetGlobalLogger(loggerInstance) 
	//GetGlobalLogger().LogInfo("Logger ready!")
	//log.Println("Logger ready!")

	// Initialize WAF
	err = InitializeWAF()
	if err != nil {
		log.Fatalf("Failed to initialize WAF: %v. ", err)
	}
	//log.Println("WAF ready!")
	//GetGlobalLogger().LogInfo("WAF ready!")

	//Initialize Proxy
	err = InitializeReverseProxy()
	if err != nil {
		log.Fatalf("Failed to initialize Reverse Proxy: %v.", err)
	}
	//log.Println("Reverse Proxy ready!")
	//GetGlobalLogger().LogInfo("Reverse Proxy ready!")
}


// The wafHandler acts as a pre-proxy WAF, inspecting and potentially blocking requests 
// before they even reach the reverseProxy for forwarding. The reverseProxy then handles
//  the actual communication with the backend application, and its ModifyResponse function 
// allows the wafHandler (specifically, the WAF transaction) to inspect the responses coming
//  from the backend before they are sent back to the client. 
func main() {
	
	// Register the wafHandler to handle all incoming HTTP requests.
	http.HandleFunc("/", WAFHandler) //  "/": This is the URL path being registered. In this case, it's the root path (/), meaning this handler will be invoked for all incoming requests to the server regardless of the path.
	port := ":8080"                  // Consider making port configurable
	log.Printf(">> Coraza Proxy listening on port %s, ready to forward to %s", port, targetAppURL)
	//GetGlobalLogger().LogInfo(">> Coraza Proxy ready!")

	// Configure server for better production readiness
	server := &http.Server{
		Addr: port,
		// Good practice: Set timeouts to avoid Slowloris attacks.
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
		Handler:      nil, // Uses http.DefaultServeMux, where wafHandler is registered
	}

	// Start the HTTP server. This call blocks until the server stops or an error occurs.
	log.Fatal(server.ListenAndServe())

	defer func() {
		if err := GetGlobalLogger().Close(); err != nil {
			log.Printf("Error closing CorazaLogger: %v", err)
		}
	}()
}