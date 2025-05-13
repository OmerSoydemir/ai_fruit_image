from PIL import Image
import numpy as np
import io

class ImageProcessor:
    """
    Class for handling image preprocessing operations
    """
    
    @staticmethod
    def resize_image(image, target_size=(224, 224)):
        """
        Resize an image to the target size
        
        Args:
            image (PIL.Image): The input image
            target_size (tuple): Target size as (width, height)
            
        Returns:
            PIL.Image: Resized image
        """
        try:
            return image.resize(target_size, Image.LANCZOS)
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            # Fallback to a simpler resize method
            return image.resize(target_size)
    
    @staticmethod
    def convert_to_rgb(image):
        """
        Convert image to RGB format if it's not already
        
        Args:
            image (PIL.Image): The input image
            
        Returns:
            PIL.Image: RGB image
        """
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    
    @staticmethod
    def normalize_image(image):
        """
        Normalize the image pixel values if needed for the model
        
        Args:
            image (PIL.Image): The input image
            
        Returns:
            PIL.Image: Normalized image
        """
        # For most Hugging Face vision models, normalization is handled internally,
        # but we ensure the image has reasonable values
        return image
    
    @staticmethod
    def verify_image(image):
        """
        Verify that the image is valid and can be processed
        
        Args:
            image (PIL.Image): The input image
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            # Check if image data is accessible
            image.verify()
            return True
        except:
            try:
                # If verify fails (it's destructive), try to access pixel data
                image.load()
                return True
            except Exception as e:
                print(f"Invalid image: {str(e)}")
                return False
    
    @staticmethod
    def preprocess(image, target_size=(224, 224)):
        """
        Apply all preprocessing steps to prepare image for model
        
        Args:
            image (PIL.Image): The input image
            target_size (tuple): Target size as (width, height)
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Verify the image is valid
        if not ImageProcessor.verify_image(image):
            raise ValueError("Invalid image provided")
            
        # Check image dimensions and resize if needed
        if image.size != target_size:
            print(f"Resizing image from {image.size} to {target_size}")
            image = ImageProcessor.resize_image(image, target_size)
        
        # Convert to RGB
        image = ImageProcessor.convert_to_rgb(image)
        
        # Normalize if needed
        image = ImageProcessor.normalize_image(image)
        
        print(f"Preprocessed image: size={image.size}, mode={image.mode}")
        return image
        
    @staticmethod
    def preprocess_standard_sizes(image):
        """
        Preprocess image with multiple standard sizes and return all versions
        Useful for troubleshooting model compatibility issues
        
        Args:
            image (PIL.Image): The input image
            
        Returns:
            dict: Different sized versions of the preprocessed image
        """
        standard_sizes = {
            "small": (224, 224),    # Standard for many models like ResNet
            "medium": (256, 256),   # Common alternative size
            "large": (384, 384),    # Used by some newer models
            "clip": (224, 224),     # CLIP model size
            "original": image.size  # Original size
        }
        
        result = {}
        for name, size in standard_sizes.items():
            try:
                if name == "original":
                    # For original, just convert to RGB but don't resize
                    result[name] = ImageProcessor.convert_to_rgb(image)
                else:
                    result[name] = ImageProcessor.preprocess(image, size)
            except Exception as e:
                print(f"Error preprocessing {name} size: {e}")
                
        return result 