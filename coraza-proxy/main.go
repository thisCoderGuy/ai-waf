// coraza-proxy/main.go
package main

import (
	"fmt"
	"log"
	"net/http"
	"rsc.io/quote" // <--- ADD THIS EXTERNAL IMPORT
)

func main() {
	// This line uses the external dependency and will force go mod tidy to record it.
	fmt.Println(quote.Hello())

	// Placeholder for your actual proxy logic
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello from Coraza AI WAF Proxy! Go version: %s\n", quote.Hello())
	})

	port := "8080"
	log.Printf("Coraza AI WAF Proxy listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}