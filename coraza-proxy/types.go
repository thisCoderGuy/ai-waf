package main



// Define a custom type for context keys to avoid collisions.
// This is a best practice for passing values through http.Request contexts.
type contextKey string
// corazaTxContextKey is used to store and retrieve the Coraza transaction from the context.
const corazaTxContextKey contextKey = "coraza-transaction"

const aiScoreContextKey contextKey = "aiScore"
const aiVerdictContextKey contextKey = "aiVerdict"

 const reqBodyContextKey contextKey = "reqBodyBytes"