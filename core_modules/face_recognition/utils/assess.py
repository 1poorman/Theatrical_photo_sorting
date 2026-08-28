import cv2
import numpy as np


def assess_face_quality_optimized(face_image):
        """
        Enhanced face quality assessment optimized for theatrical photos
        Returns:
            float: Quality score (0.0-1.0) with stricter penalties for blur/darkness
        """
        if face_image is None or face_image.size == 0:
            return 0.3  # Lower baseline for invalid images
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            height, width = gray.shape
            
            # 1. Improved Brightness Assessment (Stricter for stage photos)
            brightness = np.mean(gray)
            # Stage photos often have dramatic lighting, so we're more forgiving but still penalize extremes
            if 35 <= brightness <= 225:
                brightness_score = 1.0
            elif 20 <= brightness <= 245:
                brightness_score = 0.7
            else:
                brightness_score = 0.2  # Severe penalty for very dark/bright faces
            
            # 2. Enhanced Sharpness Assessment (Multiple blur indicators)
            # Laplacian variance (primary blur detector)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # FFT-based blur detection for more accurate assessment
            (h, w) = gray.shape
            (cX, cY) = (int(w / 2.0), int(h / 2.0))
            fft = np.fft.fft2(gray)
            fftShift = np.fft.fftshift(fft)
            
            # Create mask with radius for low-frequency removal
            mask = np.ones((h, w), dtype=np.uint8)
            cv2.circle(mask, (cX, cY), int(w/8), 0, -1)
            
            # Apply mask and compute magnitude spectrum
            fftShift *= mask
            fftMag = 20 * np.log(np.abs(fftShift) + 1e-8)
            fft_score = np.mean(fftMag)
            
            # Combined blur assessment
            if laplacian_var > 150 and fft_score > 5.0:
                sharpness_score = 1.0  # In focus
            elif laplacian_var > 100 and fft_score > 4.0:
                sharpness_score = 0.75   # Moderately in focus
            elif laplacian_var > 50 and fft_score > 3.0:
                sharpness_score = 0.4   # Slightly out of focus
            else:
                sharpness_score = 0.2   # Severely blurred/out of focus
            
            # 3. Contrast Assessment (Important for theatrical photos)
            contrast = gray.std()
            if contrast > 50:
                contrast_score = 1.0
            elif contrast > 40:
                contrast_score = 0.9
            elif contrast > 30:
                contrast_score = 0.7
            elif contrast > 20:
                contrast_score = 0.4
            else:
                contrast_score = 0.2  # Very flat image
            
            # 4. Noise detection (Stage photos may have grain)
            # Simple noise estimation using local variance
            kernel = np.ones((5,5), np.float32)/25
            smoothed = cv2.filter2D(gray, -1, kernel)
            noise = np.mean((gray - smoothed) ** 2)
            
            if noise < 100:
                noise_score = 1.0
            elif noise < 150:
                noise_score = 0.9
            elif noise < 200:
                noise_score = 0.75
            elif noise < 400:
                noise_score = 0.4
            else:
                noise_score = 0.3  # High noise likely means poor quality
            
            # 5. Weighted combination with emphasis on sharpness for theatrical context
            # In theater photography, focus is critical for character identification
            quality_score = (
                0.2 * brightness_score +
                0.4 * sharpness_score +    # Highest weight for sharpness
                0.1 * contrast_score +
                0.3 * noise_score
            )
            
            # Additional penalty for extremely low sharpness (common with out-of-focus background characters)
            if sharpness_score < 0.3:
                quality_score *= 0.7  # Extra penalty for very blurry faces
            
            return max(0.1, min(1.0, quality_score))  # Stricter bounds
            
        except Exception as e:
            print(f"Enhanced face quality assessment error: {e}")
            return 0.3  # Lower fallback score

def assess_face_quality_simple(face_image):
        """
        简化版人脸质量评估
        Returns:
            float: 质量分数 (0.0-1.0)
        """
        if face_image is None or face_image.size == 0:
            return 0.5
        
        try:
            # 转换为灰度图
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 1. 亮度评估（更宽松）
            brightness = np.mean(gray)
            if brightness < 30 or brightness > 225:
                brightness_score = 0.3
            elif brightness < 50 or brightness > 200:
                brightness_score = 0.6
            elif brightness < 70 or brightness > 180:
                brightness_score = 0.8
            else:
                brightness_score = 1.0
            
            # 2. 简单清晰度评估
            try:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 50:
                    sharpness_score = 0.3
                elif laplacian_var < 100:
                    sharpness_score = 0.6
                elif laplacian_var < 200:
                    sharpness_score = 0.8
                else:
                    sharpness_score = 1.0
            except:
                sharpness_score = 0.7
            
            # 综合质量分数（简单平均）
            quality_score = (brightness_score + sharpness_score) / 2
            
            return min(1.0, max(0.3, quality_score))  # 确保最低0.3
        
        except Exception as e:
            print(f"Face quality assessment error: {e}")
            return 0.5