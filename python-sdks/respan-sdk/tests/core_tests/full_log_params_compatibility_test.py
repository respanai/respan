"""
Test compatibility between RespanTextLogParams and RespanFullLogParams.
This ensures that any log that validates through RespanTextLogParams
will also validate through RespanFullLogParams.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from respan_sdk.respan_types.param_types import RespanTextLogParams
from respan_sdk.respan_types.log_types import RespanFullLogParams
from pydantic import ValidationError

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    # Mock pytest.raises for standalone running
    class MockPytestRaises:
        def __init__(self, exception_type):
            self.exception_type = exception_type
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exception_type.__name__} but no exception was raised")
            if not issubclass(exc_type, self.exception_type):
                raise AssertionError(f"Expected {self.exception_type.__name__} but got {exc_type.__name__}")
            return True  # Suppress the exception
    
    class MockPytest:
        @staticmethod
        def raises(exception_type):
            return MockPytestRaises(exception_type)
    
    pytest = MockPytest()


class TestFullLogParamsCompatibility:
    """Test that RespanFullLogParams provides complete backward compatibility with RespanTextLogParams."""
    
    def get_comprehensive_test_data(self):
        """Create comprehensive test data that exercises many fields."""
        return {
            # Basic LLM parameters
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["END"],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto",
            
            # Respan specific fields
            "custom_identifier": "test_123",
            "environment": "test",
            "api_key": "test_key",
            "user_id": "user_123",
            "user_email": "test@example.com",
            "organization_id": "org_123",
            "organization_name": "Test Org",
            
            # Logging fields
            "input": "Test input",
            "output": "Test output",
            "ideal_output": "Ideal output",
            "metadata": {"test_key": "test_value"},
            "note": "Test note",
            "category": "test_category",
            
            # Timing fields
            "start_time": "2024-01-01T00:00:00Z",
            "timestamp": "2024-01-01T00:01:00Z",
            "generation_time": 1.5,
            "latency": 2.0,
            "ttft": 0.5,
            
            # Tracing fields
            "trace_unique_id": "trace_123",
            "trace_name": "test_trace",
            "span_unique_id": "span_123", 
            "span_name": "test_span",
            "session_identifier": "session_123",
            
            # Usage fields
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "prompt_tokens": 10,
            "completion_tokens": 20,
            
            # Cache fields
            "cache_enabled": True,
            "cache_ttl": 3600,
            "cache_hit": False,
            
            # Cost fields
            "cost": 0.001,
            "prompt_unit_price": 0.00001,
            "completion_unit_price": 0.00002,
            
            # Embedding fields
            "embedding": [0.1, 0.2, 0.3],
            "provider_id": "openai",
            
            # Audio fields
            "audio_input_file": "input.wav",
            "audio_output_file": "output.wav",
            
            # Internal fields
            "is_test": True,
            "disable_log": False,
            "log_type": "text",
            "status": "completed",
            "error_message": None,
            
            # Advanced fields
            "customer_identifier": "customer_123",
            "customer_email": "customer@example.com",
            "group_identifier": "group_123",
            "evaluation_identifier": "eval_123",
            
            # Proxy options
            "exclude_models": ["bad_model"],
            
            # Dataset fields
            "dataset_id": "dataset_123",
            
            # Custom properties
            "metadata_indexed_string_1": "indexed_value_1",
            "metadata_indexed_numerical_1": 42.0,
        }
    
    def test_basic_compatibility(self):
        """Test basic compatibility with simple data."""
        test_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "custom_identifier": "test_basic",
        }
        
        # Both should validate successfully
        text_log_params = RespanTextLogParams(**test_data)
        full_log_params = RespanFullLogParams(**test_data)
        
        # Check that key fields match
        assert text_log_params.model == full_log_params.model
        assert text_log_params.temperature == full_log_params.temperature
        assert text_log_params.custom_identifier == full_log_params.custom_identifier
        assert len(text_log_params.messages) == len(full_log_params.messages)
        
    def test_comprehensive_compatibility(self):
        """Test compatibility with comprehensive data covering many fields."""
        test_data = self.get_comprehensive_test_data()
        
        # Both should validate successfully
        text_log_params = RespanTextLogParams(**test_data)
        full_log_params = RespanFullLogParams(**test_data)
        
        # Verify key fields are preserved
        assert text_log_params.model == full_log_params.model
        assert text_log_params.custom_identifier == full_log_params.custom_identifier
        assert text_log_params.environment == full_log_params.environment
        assert text_log_params.temperature == full_log_params.temperature
        assert text_log_params.trace_unique_id == full_log_params.trace_unique_id
        assert text_log_params.cache_enabled == full_log_params.cache_enabled
        assert text_log_params.cost == full_log_params.cost
        
    def test_field_count_compatibility(self):
        """Test that both types have the same number of fields."""
        text_log_fields = set(RespanTextLogParams.model_fields.keys())
        full_log_fields = set(RespanFullLogParams.model_fields.keys())
        
        # Should have exactly the same fields
        assert len(text_log_fields) == len(full_log_fields)
        assert text_log_fields == full_log_fields
        
    def test_validation_equivalence(self):
        """Test that validation behavior is equivalent between both types."""
        test_data = self.get_comprehensive_test_data()
        
        # Test valid data - both should succeed
        text_log_params = RespanTextLogParams(**test_data)
        full_log_params = RespanFullLogParams(**test_data)
        assert text_log_params is not None
        assert full_log_params is not None
        
        # Test invalid data - both should fail in the same way
        invalid_data = test_data.copy()
        invalid_data["temperature"] = "invalid_temperature"  # Should be float
        
        with pytest.raises(ValidationError):
            RespanTextLogParams(**invalid_data)
            
        with pytest.raises(ValidationError):
            RespanFullLogParams(**invalid_data)
    
    def test_serialization_compatibility(self):
        """Test that serialized output is compatible."""
        test_data = self.get_comprehensive_test_data()
        
        text_log_params = RespanTextLogParams(**test_data)
        full_log_params = RespanFullLogParams(**test_data)
        
        # Both should serialize to dicts
        text_dict = text_log_params.model_dump()
        full_dict = full_log_params.model_dump()
        
        # Key fields should match
        for key in ["model", "temperature", "custom_identifier", "environment"]:
            assert text_dict.get(key) == full_dict.get(key)
            
    def test_inheritance_relationship(self):
        """Test that RespanFullLogParams properly inherits from RespanLogParams."""
        from respan_sdk.respan_types.log_types import RespanLogParams
        
        # RespanFullLogParams should be a subclass of RespanLogParams
        assert issubclass(RespanFullLogParams, RespanLogParams)
        
        # Create instance and verify inheritance
        test_data = {"custom_identifier": "inheritance_test"}
        full_log_params = RespanFullLogParams(**test_data)
        
        # Should be instance of both classes
        assert isinstance(full_log_params, RespanFullLogParams)
        assert isinstance(full_log_params, RespanLogParams)
    
    def test_real_world_scenario(self):
        """Test a real-world scenario with actual LLM request/response data."""
        real_world_data = {
            # LLM request
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "stream": False,
            
            # Logging metadata
            "custom_identifier": "geography_question_001",
            "environment": "production",
            "user_id": "user_456",
            "customer_identifier": "customer_789",
            
            # Response tracking
            "prompt_tokens": 25,
            "completion_tokens": 8,
            "total_tokens": 33,
            "generation_time": 0.8,
            "cost": 0.000033,
            
            # Tracing
            "trace_unique_id": "trace_real_world_001",
            "span_name": "geography_qa",
            "session_identifier": "session_real_001",
            
            # Evaluation
            "note": "Geography question answered correctly",
            "category": "factual_qa",
            "metadata": {
                "topic": "geography",
                "difficulty": "easy",
                "correct": True
            }
        }
        
        # Both should handle real-world data identically
        text_log_params = RespanTextLogParams(**real_world_data)
        full_log_params = RespanFullLogParams(**real_world_data)
        
        # Verify critical fields match
        assert text_log_params.model == full_log_params.model == "gpt-3.5-turbo"
        assert text_log_params.custom_identifier == full_log_params.custom_identifier == "geography_question_001"
        assert text_log_params.cost == full_log_params.cost == 0.000033
        assert text_log_params.metadata == full_log_params.metadata
        
        # Both should serialize successfully
        text_serialized = text_log_params.model_dump(exclude_none=True)
        full_serialized = full_log_params.model_dump(exclude_none=True)
        
        # Should have the same keys in serialized form
        assert set(text_serialized.keys()) == set(full_serialized.keys())


if __name__ == "__main__":
    # Run the tests
    test_instance = TestFullLogParamsCompatibility()
    
    print("Running RespanFullLogParams compatibility tests...")
    
    try:
        test_instance.test_basic_compatibility()
        print("✅ Basic compatibility test passed")
        
        test_instance.test_comprehensive_compatibility()
        print("✅ Comprehensive compatibility test passed")
        
        test_instance.test_field_count_compatibility()
        print("✅ Field count compatibility test passed")
        
        test_instance.test_validation_equivalence()
        print("✅ Validation equivalence test passed")
        
        test_instance.test_serialization_compatibility()
        print("✅ Serialization compatibility test passed")
        
        test_instance.test_inheritance_relationship()
        print("✅ Inheritance relationship test passed")
        
        test_instance.test_real_world_scenario()
        print("✅ Real-world scenario test passed")
        
        print("\n🎉 All compatibility tests passed! RespanFullLogParams is fully backward compatible.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()