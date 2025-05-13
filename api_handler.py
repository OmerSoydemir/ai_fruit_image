import requests
import io
import os
import json
import base64
from dotenv import load_dotenv

class APIHandler:
    """
    Class for handling API requests to image classification services
    """
    
    def __init__(self, model_id="microsoft/resnet-50", api_token=None):
        """
        Initialize the API handler
        
        Args:
            model_id (str): The Hugging Face model ID to use
            api_token (str): Hugging Face API token, if provided directly
        """
        # Load environment variables
        load_dotenv('env.example')
        
        # Get API token from parameter, environment variable, or default token
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN") or ""
        
        # Set the API URL based on the model ID
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        print(f"API URL: {self.api_url}")
        print(f"Using token: {self.api_token[:5]}...{self.api_token[-4:]}" if len(self.api_token) > 10 else "Token too short or empty")
    
    def classify_image(self, image):
        """
        Send an image to the Hugging Face API for classification
        
        Args:
            image (PIL.Image): The preprocessed image to classify
            
        Returns:
            dict: Classification results or error message
        """
        # Check if token is configured
        if not self.is_configured():
            return {"error": "API token is not configured correctly"}
            
        # First, try with direct binary data (standard approach)
        result = self._try_binary_upload(image)
        if "error" not in result:
            return result
            
        # If binary upload fails, try with base64 encoding
        print("Binary upload failed, trying base64 encoding...")
        result = self._try_base64_upload(image)
        if "error" not in result:
            return result
            
        # If base64 encoding fails, try simple base64
        print("Base64 upload failed, trying simple base64...")
        result = self._try_simple_base64(image)
        if "error" not in result:
            return result
            
        # If base64 encoding fails, try form upload
        print("Simple base64 upload failed, trying form upload...")
        result = self._try_form_upload(image)
        return result
    
    def _try_binary_upload(self, image):
        """Try uploading image as binary data"""
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Set headers with authorization
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/octet-stream"
            }
            
            print(f"Sending binary request to: {self.api_url}")
            
            # Make API request
            response = requests.post(self.api_url, headers=headers, data=img_byte_arr)
            
            print(f"Binary response status code: {response.status_code}")
            
            return self._process_response(response)
        except Exception as e:
            return {"error": f"Binary upload failed: {str(e)}"}
    
    def _try_base64_upload(self, image):
        """Try uploading image as base64 encoded JSON"""
        try:
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Set headers with authorization
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Create payload with base64 image
            payload = {
                "inputs": {
                    "image": encoded_img
                }
            }
            
            print(f"Sending base64 request to: {self.api_url}")
            
            # Make API request
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            print(f"Base64 response status code: {response.status_code}")
            
            return self._process_response(response)
        except Exception as e:
            return {"error": f"Base64 upload failed: {str(e)}"}
    
    def _try_simple_base64(self, image):
        """Try uploading image as simple base64 string (alternate format)"""
        try:
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Set headers with authorization
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Create simpler payload (just the base64 string)
            payload = {"inputs": encoded_img}
            
            print(f"Sending simple base64 request to: {self.api_url}")
            
            # Make API request
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            print(f"Simple base64 response status code: {response.status_code}")
            
            return self._process_response(response)
        except Exception as e:
            return {"error": f"Simple base64 upload failed: {str(e)}"}
    
    def _try_form_upload(self, image):
        """Try uploading image as multipart form data"""
        try:
            # Convert image to bytes for form upload
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Set headers with authorization
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            # Create form data
            files = {
                'file': ('image.jpg', img_byte_arr, 'image/jpeg')
            }
            
            print(f"Sending form request to: {self.api_url}")
            
            # Make API request
            response = requests.post(self.api_url, headers=headers, files=files)
            
            print(f"Form response status code: {response.status_code}")
            
            return self._process_response(response)
        except Exception as e:
            return {"error": f"Form upload failed: {str(e)}"}
    
    def _process_response(self, response):
        """Process API response and handle errors"""
        # Check for successful response
        if response.status_code != 200:
            try:
                error_detail = response.json()
            except json.JSONDecodeError:
                error_detail = response.text[:200] + "..." if len(response.text) > 200 else response.text
            return {"error": f"API Error: {error_detail}"}
        
        # Parse and return results
        try:
            results = response.json()
            
            # Verify the results have the expected format
            if not isinstance(results, list) and isinstance(results, dict):
                # Some models return a dict with prediction info
                if "error" in results:
                    return {"error": f"API Error: {results['error']}"}
                # Try to convert to compatible format
                return [results]
            elif not isinstance(results, list):
                return {"error": f"Unexpected API response format: {results}"}
                
            # Validate each item in the results
            validated_results = []
            for item in results:
                if isinstance(item, dict) and "label" in item and item["label"] is not None:
                    validated_results.append(item)
                else:
                    print(f"Skipping invalid result item: {item}")
                    
            if not validated_results:
                return {"error": "API returned no valid classification results"}
                
            return validated_results
            
        except json.JSONDecodeError:
            return {"error": f"Failed to parse API response: {response.text[:100]}..."}
    
    def is_configured(self):
        """
        Check if the API is properly configured with a token
        
        Returns:
            bool: True if API token is set, False otherwise
        """
        return self.api_token is not None and len(self.api_token) > 10 and self.api_token.startswith("hf_") 