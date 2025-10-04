#!/usr/bin/env python3
"""
Script to test the AWS deployment of CIFAR-10 CNN project.
This script performs comprehensive health checks and functionality tests.
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

class DeploymentTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.mlflow_url = f"{self.base_url}/mlflow"
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_api_health(self) -> bool:
        """Test API health endpoint."""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health: {data}")
                return data.get('status') == 'healthy'
            else:
                print(f"❌ API Health failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Health error: {e}")
            return False
    
    def test_mlflow_health(self) -> bool:
        """Test MLflow health endpoint."""
        try:
            response = self.session.get(f"{self.mlflow_url}/health")
            if response.status_code == 200:
                print("✅ MLflow Health: OK")
                return True
            else:
                print(f"❌ MLflow Health failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ MLflow Health error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint."""
        try:
            response = self.session.get(f"{self.api_url}/model-info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model Info: {data['model_name']} with {data['parameters']:,} parameters")
                return True
            else:
                print(f"❌ Model Info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model Info error: {e}")
            return False
    
    def test_random_images(self) -> bool:
        """Test random images endpoint."""
        try:
            response = self.session.get(f"{self.api_url}/random-images")
            if response.status_code == 200:
                data = response.json()
                images = data.get('images', [])
                print(f"✅ Random Images: Retrieved {len(images)} images")
                return len(images) > 0
            else:
                print(f"❌ Random Images failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Random Images error: {e}")
            return False
    
    def test_prediction(self) -> bool:
        """Test prediction endpoint."""
        try:
            # First get some images
            response = self.session.get(f"{self.api_url}/random-images")
            if response.status_code != 200:
                print("❌ Prediction test failed: Could not get images")
                return False
            
            data = response.json()
            images = data.get('images', [])
            if not images:
                print("❌ Prediction test failed: No images available")
                return False
            
            # Test prediction on first image
            image_index = images[0]['index']
            prediction_data = {"image_index": image_index}
            
            response = self.session.post(
                f"{self.api_url}/predict",
                json=prediction_data
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted_class = data.get('predicted_class')
                confidence = data.get('confidence')
                print(f"✅ Prediction: {predicted_class} (confidence: {confidence:.3f})")
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_mlflow_ui(self) -> bool:
        """Test MLflow UI accessibility."""
        try:
            response = self.session.get(f"{self.mlflow_url}/")
            if response.status_code == 200:
                print("✅ MLflow UI: Accessible")
                return True
            else:
                print(f"❌ MLflow UI failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ MLflow UI error: {e}")
            return False
    
    def test_performance(self) -> Dict[str, float]:
        """Test API performance."""
        performance_results = {}
        
        # Test API health response time
        start_time = time.time()
        if self.test_api_health():
            performance_results['api_health_time'] = time.time() - start_time
        
        # Test prediction response time
        start_time = time.time()
        if self.test_prediction():
            performance_results['prediction_time'] = time.time() - start_time
        
        return performance_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🧪 Starting deployment tests...")
        print(f"🌐 Base URL: {self.base_url}")
        print("=" * 60)
        
        results = {
            'api_health': False,
            'mlflow_health': False,
            'model_info': False,
            'random_images': False,
            'prediction': False,
            'mlflow_ui': False,
            'performance': {}
        }
        
        # Run health checks
        results['api_health'] = self.test_api_health()
        results['mlflow_health'] = self.test_mlflow_health()
        
        # Run functionality tests
        results['model_info'] = self.test_model_info()
        results['random_images'] = self.test_random_images()
        results['prediction'] = self.test_prediction()
        results['mlflow_ui'] = self.test_mlflow_ui()
        
        # Run performance tests
        results['performance'] = self.test_performance()
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 6
        passed_tests = sum([
            results['api_health'],
            results['mlflow_health'],
            results['model_info'],
            results['random_images'],
            results['prediction'],
            results['mlflow_ui']
        ])
        
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
        
        if results['performance']:
            print("\n⏱️  Performance:")
            for test, time_taken in results['performance'].items():
                print(f"   {test}: {time_taken:.2f}s")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! Deployment is working correctly.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the deployment.")
            return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test AWS deployment')
    parser.add_argument('--url', type=str, required=True,
                       help='Base URL of the deployment (e.g., http://alb-dns-name)')
    parser.add_argument('--wait', type=int, default=0,
                       help='Wait time in seconds before starting tests')
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for services to start...")
        time.sleep(args.wait)
    
    # Create tester and run tests
    tester = DeploymentTester(args.url)
    results = tester.run_all_tests()
    
    # Print summary
    success = tester.print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

