from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any

def extract_metadata(image_path: str) -> Dict[str, Any]:
    """Extracts EXIF and basic image metadata."""
    metadata = {}
    try:
        with Image.open(image_path) as img:
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['size'] = img.size
            
            exif_data = img.getexif()
            if exif_data:
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    # Filter out large binary chunks (e.g., MakerNote)
                    if isinstance(value, bytes):
                        continue
                    exif_dict[tag] = str(value)
                metadata['exif'] = exif_dict
                
                # Check for software/processing signatures often indicative of AI or editing
                software = exif_dict.get('Software', '').lower()
                ai_keywords = ['midjourney', 'dall-e', 'stable diffusion', 'comfyui', 'automatic1111']
                metadata['ai_software_detected'] = any(kw in software for kw in ai_keywords)
            else:
                metadata['exif'] = None
                metadata['ai_software_detected'] = False
                
            # Note: A real C2PA implementation would use the c2pa-rs python bindings.
            # For hackathon purposes without rust tooling, we return a placeholder.
            metadata['c2pa_detected'] = False
            
    except Exception as e:
        metadata['error'] = str(e)
        
    return metadata
