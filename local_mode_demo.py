#!/usr/bin/env python3
"""
SageMaker Local Mode Demo

This script demonstrates how to use SageMaker Python SDK in local mode.
Local mode runs training and inference locally using Docker containers.

Prerequisites:
- pip install 'sagemaker[local]' --upgrade
- Docker and Docker Compose V2 installed
"""

import os
import numpy as np
from sagemaker.sklearn import SKLearn
from sagemaker.local import LocalSession
import boto3

def create_training_script():
    """Create a simple training script for the demo"""
    training_script = '''
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    
    args = parser.parse_args()
    
    # Create simple synthetic data for demo
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
'''
    
    os.makedirs("code", exist_ok=True)
    with open("code/train.py", "w") as f:
        f.write(training_script)
    print("✓ Created training script at code/train.py")

def demo_local_training():
    """Demonstrate local mode training"""
    print("\n=== Local Mode Training Demo ===")
    
    # Set environment variables for AWS configuration
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
    
    # Create local session with explicit region
    local_session = LocalSession(boto_session=boto3.Session(region_name='us-west-2'))
    local_session.config = {'local': {'local_code': True}}
    
    # Create SKLearn estimator with local instance type
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        source_dir='code',
        role='arn:aws:iam::123456789012:role/SageMakerRole',  # Dummy role for local mode
        instance_type='local',  # This triggers local mode
        instance_count=1,
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=local_session
    )
    
    print("Starting local training...")
    # Train locally (no S3 data needed in local mode)
    sklearn_estimator.fit()
    
    print("✓ Local training completed!")
    return sklearn_estimator

def demo_local_inference(estimator):
    """Demonstrate local mode inference"""
    print("\n=== Local Mode Inference Demo ===")
    
    # Deploy model locally
    print("Deploying model to local endpoint...")
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type='local'  # This creates a local endpoint
    )
    
    # Create test data
    test_data = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(test_data)
    print(f"Predictions: {predictions}")
    
    # Clean up
    print("Cleaning up local endpoint...")
    try:
        predictor.delete_endpoint()
        predictor.delete_model()
        print("✓ Cleanup completed")
    except Exception as cleanup_error:
        print(f"Note: Cleanup had minor issue (this is normal): {cleanup_error}")
    
    print("✓ Local inference demo completed!")

def test_docker_host_function():
    """Test the get_docker_host function you modified"""
    try:
        import sys
        sys.path.insert(0, 'src')
        from sagemaker.local.utils import get_docker_host
        
        docker_host = get_docker_host()
        print(f"✓ get_docker_host() returns: {docker_host}")
        
        if docker_host != "localhost":
            print("  → Your rootless Docker fix is active!")
        else:
            print("  → Using traditional Docker setup")
            
        return docker_host
    except Exception as e:
        print(f"Note: Could not test get_docker_host(): {e}")
        return None

def check_aws_auth():
    """Check if AWS credentials are valid"""
    try:
        import boto3
        # Set region first
        os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
        sts = boto3.client('sts', region_name='us-west-2')
        sts.get_caller_identity()
        return True
    except Exception as e:
        print(f"❌ AWS authentication failed: {e}")
        return False

def main():
    """Run the complete local mode demo"""
    print("SageMaker Local Mode Demo")
    print("=" * 50)
    
    # Test your Docker host function
    print("Testing your get_docker_host() fix:")
    test_docker_host_function()
    
    # Check AWS authentication first
    if not check_aws_auth():
        print("\n🔧 TO FIX:")
        print("1. Refresh AWS credentials:")
        print("   aws sso login")
        print("   # or however you normally authenticate")
        print("\n2. Verify credentials work:")
        print("   aws sts get-caller-identity")
        print("\n3. Then run this demo again")
        print("\n📝 NOTE: Local mode needs AWS credentials to pull Docker images")
        return
    
    try:
        # Step 1: Create training script
        create_training_script()
        
        # Step 2: Run local training
        estimator = demo_local_training()
        
        # Step 3: Run local inference
        demo_local_inference(estimator)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey points about Local Mode:")
        print("- Uses Docker containers locally instead of AWS infrastructure")
        print("- Great for testing before deploying to SageMaker")
        print("- Supports all major ML frameworks")
        print("- Can use local file system instead of S3")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\nCommon issues:")
        print("- AWS credentials expired: run 'aws sso login'")
        print("- Docker not running: start Docker")
        print("- Missing dependencies: pip install 'sagemaker[local]' --upgrade")

if __name__ == "__main__":
    main()