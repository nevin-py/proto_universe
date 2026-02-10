use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Python bindings for ProtoGalaxy with variable k support
/// 
/// This module provides a simplified interface to the ProtoGalaxy folding scheme
/// from the sonobe library, with support for multi-instance folding.
#[pymodule]
fn sonobe_protogalaxy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProtoGalaxy>()?;
    Ok(())
}

/// Python wrapper for ProtoGalaxy folding scheme
/// 
/// This class provides a simplified interface to ProtoGalaxy, supporting
/// variable k (number of incoming instances) for multi-instance folding.
#[pyclass]
struct PyProtoGalaxy {
    k: usize,
}

#[pymethods]
impl PyProtoGalaxy {
    /// Create a new ProtoGalaxy instance with the specified k value
    /// 
    /// Args:
    ///     k (int): Number of incoming instances to fold. Must satisfy:
    ///              - k >= 1
    ///              - k + 1 must be a power of two
    /// 
    /// Returns:
    ///     PyProtoGalaxy: A new ProtoGalaxy instance
    /// 
    /// Raises:
    ///     ValueError: If k is invalid (< 1 or k+1 is not a power of two)
    /// 
    /// Examples:
    ///     >>> pg = PyProtoGalaxy(k=1)  # Single instance folding
    ///     >>> pg = PyProtoGalaxy(k=3)  # Multi-instance folding (4 total instances)
    #[new]
    fn new(k: usize) -> PyResult<Self> {
        // Validate k
        if k < 1 {
            return Err(PyValueError::new_err(format!(
                "Invalid k value: {}. k must be at least 1",
                k
            )));
        }
        
        // Validate k+1 is a power of two
        if !(k + 1).is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "Invalid k value: {}. k+1 ({}) must be a power of two. Valid k values: 1, 3, 7, 15, 31, ...",
                k, k + 1
            )));
        }
        
        Ok(PyProtoGalaxy { k })
    }
    
    /// Get the k value (number of incoming instances)
    /// 
    /// Returns:
    ///     int: The k value for this ProtoGalaxy instance
    #[getter]
    fn get_k(&self) -> usize {
        self.k
    }
    
    /// Get the total number of instances (k + 1)
    /// 
    /// Returns:
    ///     int: Total number of instances that will be folded
    fn total_instances(&self) -> usize {
        self.k + 1
    }
    
    /// Check if the configuration is valid
    /// 
    /// Returns:
    ///     bool: True if the configuration is valid
    fn is_valid(&self) -> bool {
        self.k >= 1 && (self.k + 1).is_power_of_two()
    }
    
    /// Get a string representation of the ProtoGalaxy instance
    fn __repr__(&self) -> String {
        format!(
            "PyProtoGalaxy(k={}, total_instances={})",
            self.k,
            self.k + 1
        )
    }
    
    /// Preprocess step (simplified interface)
    /// 
    /// Note: Full preprocessing requires R1CS circuit setup, which is complex
    /// to expose via Python. This is a placeholder for the simplified interface.
    /// 
    /// Returns:
    ///     str: Status message
    fn preprocess(&self) -> PyResult<String> {
        Ok(format!(
            "Preprocessing with k={} (total {} instances). Full implementation requires circuit setup.",
            self.k, self.k + 1
        ))
    }
    
    /// Prove step (simplified interface)
    /// 
    /// Note: Full prove_step requires witness and instance data, which is complex
    /// to expose via Python. This is a placeholder for the simplified interface.
    /// 
    /// Returns:
    ///     str: Status message
    fn prove_step(&self) -> PyResult<String> {
        Ok(format!(
            "Proving with k={} incoming instances. Full implementation requires witness data.",
            self.k
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_k_values() {
        // Valid k values: 1, 3, 7, 15, 31, ... (k+1 is power of 2)
        assert!(PyProtoGalaxy::new(1).is_ok());
        assert!(PyProtoGalaxy::new(3).is_ok());
        assert!(PyProtoGalaxy::new(7).is_ok());
        assert!(PyProtoGalaxy::new(15).is_ok());
    }

    #[test]
    fn test_invalid_k_values() {
        // Invalid k values
        assert!(PyProtoGalaxy::new(0).is_err());  // k < 1
        assert!(PyProtoGalaxy::new(2).is_err());  // k+1 = 3 (not power of 2)
        assert!(PyProtoGalaxy::new(4).is_err());  // k+1 = 5 (not power of 2)
        assert!(PyProtoGalaxy::new(5).is_err());  // k+1 = 6 (not power of 2)
    }

    #[test]
    fn test_total_instances() {
        let pg = PyProtoGalaxy::new(3).unwrap();
        assert_eq!(pg.total_instances(), 4);
    }
}
