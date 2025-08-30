#!/usr/bin/env python3
"""
Script to verify that animal images are properly accessible
"""

import os
from pathlib import Path

def verify_images():
    """Verify that all animal images exist and are accessible"""
    print("🔍 Verifying animal images...")
    
    image_dir = Path("static/images")
    animals = ['cat', 'dog', 'elephant']
    
    for animal in animals:
        jpg_path = image_dir / f"{animal}.jpg"
        jpeg_path = image_dir / f"{animal}.jpeg"
        
        print(f"\n🐾 {animal.upper()}:")
        
        if jpg_path.exists():
            size = jpg_path.stat().st_size
            print(f"  ✅ {jpg_path.name} - {size} bytes")
        else:
            print(f"  ❌ {jpg_path.name} - NOT FOUND")
            
        if jpeg_path.exists():
            size = jpeg_path.stat().st_size
            print(f"  ✅ {jpeg_path.name} - {size} bytes")
        else:
            print(f"  ❌ {jpeg_path.name} - NOT FOUND")
    
    print(f"\n📁 Image directory: {image_dir.absolute()}")
    print(f"📊 Total files in images directory: {len(list(image_dir.glob('*')))}")

if __name__ == '__main__':
    verify_images()
