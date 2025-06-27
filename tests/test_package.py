# Step Detection Package Tests

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.step_detection.models.model_utils import create_cnn_model
from src.step_detection.utils.data_processor import load_step_data


def test_data_loading():
    """Test data loading functionality."""
    # This would require sample data to be present
    pass


def test_model_creation():
    """Test model creation."""
    model = create_cnn_model()
    assert model is not None
    assert len(model.layers) > 0
    print("✅ Model creation test passed!")


def test_package_import():
    """Test package imports."""
    try:
        import src.step_detection

        print("✅ Package import test passed!")
        return True
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Step Detection Package Tests...")

    test_package_import()
    test_model_creation()

    print("✅ All tests completed!")
