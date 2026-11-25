import { 
  startTracing, 
  getCurrentSpan, 
  updateCurrentSpan, 
  addSpanEvent, 
  recordSpanException,
  setSpanStatus,
  withManualSpan 
} from "../dist/utils/tracing";
import { withWorkflow, withTask, withAgent } from "../dist/decorators/base";
import { SpanStatusCode } from "@opentelemetry/api";

// Initialize tracing
startTracing({
  appName: "advanced-respan-example",
  apiKey: process.env.RESPAN_API_KEY || "test-key",
  baseURL: process.env.RESPAN_BASE_URL || "https://api.respan.ai",
  traceContent: true,
  logLevel: "info"
});

// Example 1: Advanced span updating with proper Respan parameters
async function advancedLLMCall() {
  console.log("\n=== Example 1: Advanced LLM Call with Respan Parameters ===");
  
  return withAgent({
    name: "smartAgent",
    associationProperties: {
      userId: "user123",
      sessionId: "session456",
      requestId: "req-789"
    }
  }, async () => {
    // Update span with comprehensive Respan parameters
    updateCurrentSpan({
      respanParams: {
        // Model configuration
        model: "gpt-4",
        provider: "openai",
        temperature: 0.7,
        max_tokens: 1000,
        top_p: 0.9,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        
        // User identification
        user_id: "user123",
        customer_identifier: "customer456",
        customer_email: "user@example.com",
        customer_name: "John Doe",
        
        // Session and threading
        thread_identifier: "thread789",
        trace_group_identifier: "group123",
        
        // Custom metadata
        metadata: {
          experiment: "A/B-test-v2",
          feature_flag: "new_ui_enabled",
          model_version: "v2.1",
          deployment: "production"
        },
        
        // Evaluation
        for_eval: true,
        
        // Environment
        environment: "production",
        is_test: false
      },
      attributes: {
        "custom.operation": "llm_generation",
        "custom.priority": "high",
        "custom.retry_count": 0
      }
    });

    // Simulate processing stages
    addSpanEvent("processing_started", {
      "stage": "input_validation",
      "input_length": 150
    });

    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Update span during processing
    updateCurrentSpan({
      name: "smartAgent.generating",
      attributes: {
        "processing.stage": "llm_call",
        "processing.tokens_requested": 1000
      }
    });

    addSpanEvent("llm_call_initiated", {
      "model": "gpt-4",
      "temperature": 0.7
    });

    await new Promise(resolve => setTimeout(resolve, 200));

    // Update with completion information
    updateCurrentSpan({
      respanParams: {
        completion_tokens: 245,
        prompt_tokens: 150,
        total_tokens: 395,
        cost: 0.0234,
        latency: 1.2,
        tokens_per_second: 204.2
      },
      status: SpanStatusCode.OK,
      statusDescription: "LLM call completed successfully",
      attributes: {
        "processing.stage": "completed",
        "result.tokens_generated": 245,
        "result.quality_score": 0.95
      }
    });

    return {
      result: "Advanced LLM processing completed successfully",
      tokens_used: 395,
      cost: 0.0234,
      model: "gpt-4"
    };
  });
}

// Example 2: Error handling with proper span updates
async function errorHandlingExample() {
  console.log("\n=== Example 2: Error Handling with Span Updates ===");
  
  return withTask({
    name: "errorProneTask",
    associationProperties: {
      operationType: "risky_operation"
    }
  }, async () => {
    try {
      // Update span with initial parameters
      updateCurrentSpan({
        respanParams: {
          custom_identifier: "risky-op-001",
          metadata: {
            risk_level: "high",
            retry_enabled: true
          }
        },
        attributes: {
          "operation.type": "risky",
          "operation.attempt": 1
        }
      });

      // Simulate potential failure
      const shouldFail = Math.random() > 0.5;
      
      if (shouldFail) {
        const error = new Error("Simulated operation failure");
        
        // Record the exception in the span
        recordSpanException(error);
        
        // Update span with error information
        updateCurrentSpan({
          respanParams: {
            error_bit: 1,
            error_message: error.message,
            status: "failed"
          },
          status: SpanStatusCode.ERROR,
          statusDescription: "Operation failed due to simulated error",
          attributes: {
            "error.type": "simulation_error",
            "error.recoverable": true
          }
        });

        throw error;
      }

      // Success path
      updateCurrentSpan({
        respanParams: {
          status: "success"
        },
        status: SpanStatusCode.OK,
        statusDescription: "Risky operation completed successfully",
        attributes: {
          "operation.result": "success"
        }
      });

      return { result: "Operation completed successfully" };
    } catch (error) {
      // Additional error handling
      addSpanEvent("error_handled", {
        "error_message": (error as Error).message,
        "handled_at": new Date().toISOString()
      });
      
      throw error;
    }
  });
}

// Example 3: Multi-step workflow with span updates
async function multiStepWorkflow() {
  console.log("\n=== Example 3: Multi-step Workflow ===");
  
  return withWorkflow({
    name: "dataProcessingWorkflow",
    associationProperties: {
      workflowId: "wf-001",
      batchId: "batch-123"
    }
  }, async () => {
    const results: Array<{ records?: number; success_rate?: number; processed?: number; accuracy?: number }> = [];
    
    // Step 1: Data ingestion
    const ingestionResult = await withTask({
      name: "dataIngestion"
    }, async () => {
      updateCurrentSpan({
        respanParams: {
          custom_identifier: "ingestion-step",
          metadata: {
            step: "ingestion",
            data_source: "api"
          }
        },
        attributes: {
          "step.number": 1,
          "step.name": "data_ingestion"
        }
      });

      await new Promise(resolve => setTimeout(resolve, 50));
      
      updateCurrentSpan({
        attributes: {
          "ingestion.records_processed": 1000,
          "ingestion.success_rate": 0.98
        }
      });

      return { records: 1000, success_rate: 0.98 };
    });

    results.push(ingestionResult);

    // Step 2: Data processing
    const processingResult = await withTask({
      name: "dataProcessing"
    }, async () => {
      updateCurrentSpan({
        respanParams: {
          custom_identifier: "processing-step",
          metadata: {
            step: "processing",
            algorithm: "advanced_nlp"
          }
        },
        attributes: {
          "step.number": 2,
          "step.name": "data_processing",
          "input.records": ingestionResult.records
        }
      });

      await new Promise(resolve => setTimeout(resolve, 100));
      
      updateCurrentSpan({
        attributes: {
          "processing.records_processed": ingestionResult.records,
          "processing.accuracy": 0.94
        }
      });

      return { processed: ingestionResult.records, accuracy: 0.94 };
    });

    results.push(processingResult);

    // Update workflow span with final results
    updateCurrentSpan({
      respanParams: {
        metadata: {
          workflow_status: "completed",
          total_records: ingestionResult.records,
          final_accuracy: processingResult.accuracy
        }
      },
      status: SpanStatusCode.OK,
      statusDescription: "Workflow completed successfully",
      attributes: {
        "workflow.total_steps": 2,
        "workflow.success": true,
        "workflow.total_records": ingestionResult.records
      }
    });

    return {
      workflow_id: "wf-001",
      steps_completed: 2,
      total_records: ingestionResult.records,
      final_accuracy: processingResult.accuracy,
      results
    };
  });
}

// Example 4: Manual span with custom tracing
async function manualSpanExample() {
  console.log("\n=== Example 4: Manual Span Creation ===");
  
  return withManualSpan(
    "customDatabaseOperation",
    async (span) => {
      // Update the manual span with Respan parameters
      updateCurrentSpan({
        respanParams: {
          custom_identifier: "db-op-001",
          metadata: {
            operation_type: "database_query",
            database: "postgresql",
            table: "users"
          }
        },
        attributes: {
          "db.operation": "SELECT",
          "db.table": "users",
          "db.query_type": "analytical"
        }
      });

      // Simulate database operation
      addSpanEvent("query_started", {
        "query": "SELECT * FROM users WHERE active = true",
        "expected_rows": 5000
      });

      await new Promise(resolve => setTimeout(resolve, 150));

      // Update with results
      updateCurrentSpan({
        attributes: {
          "db.rows_returned": 4823,
          "db.execution_time_ms": 145,
          "db.cache_hit": false
        }
      });

      addSpanEvent("query_completed", {
        "rows_returned": 4823,
        "execution_time": 145
      });

      return {
        operation: "database_query",
        rows_returned: 4823,
        execution_time_ms: 145
      };
    },
    {
      "operation.type": "database",
      "operation.manual": true
    }
  );
}

// Main execution function
async function main() {
  console.log("Starting Advanced Respan Tracing Examples...");

  try {
    // Run all examples
    const result1 = await advancedLLMCall();
    console.log("Advanced LLM call result:", result1);

    try {
      const result2 = await errorHandlingExample();
      console.log("Error handling result:", result2);
    } catch (error) {
      console.log("Expected error caught:", (error as Error).message);
    }

    const result3 = await multiStepWorkflow();
    console.log("Multi-step workflow result:", result3);

    const result4 = await manualSpanExample();
    console.log("Manual span result:", result4);

    console.log("\n✅ All examples completed successfully!");
    console.log("Check your Respan dashboard for detailed traces with proper attributes and metadata.");

  } catch (error) {
    console.error("❌ Example execution failed:", error);
  }
}

// Run the examples
main().catch(console.error);

export {
  advancedLLMCall,
  errorHandlingExample,
  multiStepWorkflow,
  manualSpanExample
}; 