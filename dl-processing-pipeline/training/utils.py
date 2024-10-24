from PIL import Image
from io import BytesIO
from torchvision import transforms

class DecodeJPEG:
        def __call__(self, raw_bytes):
            return Image.open(BytesIO(raw_bytes))
        
class ConditionalNormalize:
            def __init__(self, mean, std):
                self.normalize = transforms.Normalize(mean=mean, std=std)

            def __call__(self, tensor):
                # Only apply normalization if the tensor has 3 channels
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)  # Repeat the single channel across the 3 RGB channels
                
                # Apply normalization to 3-channel (RGB) images
                return self.normalize(tensor)
                

