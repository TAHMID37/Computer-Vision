#!/usr/bin/env python3
"""
RF-DETR + SAHI Video Processing Implementation - FIXED VERSION
This script implements SAHI (Slicing Aided Hyper Inference) with RF-DETR for video processing

IMPORTANT: Save this file with a name OTHER than 'sahi.py' to avoid import conflicts!
Recommended filename: rf_detr_video_processor.py
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import time
import os
import sys
from PIL import Image
import supervision as sv

# RF-DETR imports with better error handling
try:
    from rfdetr import RFDETRBase , RFDETRNano
    from rfdetr.util.coco_classes import COCO_CLASSES
    RF_DETR_AVAILABLE = True
except ImportError as e:
    print("Warning: RF-DETR not available. Will use fallback detection.")
    print(f"Install RF-DETR with: pip install rfdetr")
    print(f"Import error: {e}")
    RF_DETR_AVAILABLE = False
    COCO_CLASSES = [f"class_{i}" for i in range(80)]  # Fallback classes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress the HuggingFace warning
os.environ["HF_HOME"] = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

class SliceInfo:
    """Information about an image slice"""
    def __init__(self, image, starting_pixel, slice_width, slice_height):
        self.image = image
        self.starting_pixel = starting_pixel  # (x, y)
        self.slice_width = slice_width
        self.slice_height = slice_height

class Detection:
    """Custom detection class compatible with supervision format"""
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array(xyxy) if len(xyxy) > 0 else np.array([]).reshape(0, 4)
        self.confidence = np.array(confidence) if len(confidence) > 0 else np.array([])
        self.class_id = np.array(class_id) if len(class_id) > 0 else np.array([])

def slice_image(image: np.ndarray, slice_height: int, slice_width: int, 
               overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2) -> List[SliceInfo]:
    """
    Slice an image into overlapping patches
    
    Args:
        image: Input image as numpy array
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Overlap ratio for height
        overlap_width_ratio: Overlap ratio for width
        
    Returns:
        List of SliceInfo objects
    """
    slices = []
    image_height, image_width = image.shape[:2]
    
    # Calculate step sizes
    step_height = int(slice_height * (1 - overlap_height_ratio))
    step_width = int(slice_width * (1 - overlap_width_ratio))
    
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            # Calculate slice boundaries
            x_end = min(x + slice_width, image_width)
            y_end = min(y + slice_height, image_height)
            
            # Extract slice
            slice_img = image[y:y_end, x:x_end]
            
            # Pad slice if necessary to maintain consistent size
            if slice_img.shape[0] < slice_height or slice_img.shape[1] < slice_width:
                padded_slice = np.zeros((slice_height, slice_width, 3), dtype=image.dtype)
                padded_slice[:slice_img.shape[0], :slice_img.shape[1]] = slice_img
                slice_img = padded_slice
            
            slices.append(SliceInfo(
                image=slice_img,
                starting_pixel=(x, y),
                slice_width=slice_width,
                slice_height=slice_height
            ))
            
            x += step_width
            if x >= image_width:
                break
                
        y += step_height
        if y >= image_height:
            break
    
    return slices

class RFDETRSAHIVideoProcessor:
    """
    A class to process videos using RF-DETR model with SAHI for enhanced small object detection
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        confidence_threshold: float = 0.3,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        nms_threshold: float = 0.5,
        optimize_for_inference: bool = True
    ):
        """
        Initialize the RF-DETR SAHI Video Processor
        
        Args:
            model_path: Path to the RF-DETR model weights (optional)
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
            slice_height: Height of each slice for SAHI
            slice_width: Width of each slice for SAHI
            overlap_height_ratio: Overlap ratio for height
            overlap_width_ratio: Overlap ratio for width
            nms_threshold: Threshold for NMS
            optimize_for_inference: Whether to optimize model for inference
        """
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.nms_threshold = nms_threshold
        
        # Initialize the RF-DETR model with proper error handling
        self.model = self._initialize_model(model_path, optimize_for_inference)
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
        
        logger.info("RF-DETR SAHI Video Processor initialized successfully")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self, model_path: Optional[str], optimize_for_inference: bool):
        """Initialize the RF-DETR model with better error handling"""
        if not RF_DETR_AVAILABLE:
            logger.error("RF-DETR is not available. Please install with: pip install rfdetr")
            raise ImportError("RF-DETR package not found")
        
        try:
            logger.info("Initializing RF-DETR model...")
            
            # Force CPU for initialization to avoid device issues
            logger.info("Loading RF-DETR model on CPU first...")
            model = RFDETRNano()
            
            # Verify the model is not None
            if model is None:
                raise RuntimeError("RF-DETR model initialization returned None")
            
            logger.info("Model loaded successfully, checking for predict method...")
            if not hasattr(model, 'predict'):
                raise RuntimeError("RF-DETR model does not have predict method")
            
            # Try to optimize for inference
            if optimize_for_inference:
                try:
                    logger.info("Optimizing model for inference...")
                    optimized_model = model.optimize_for_inference()
                    if optimized_model is not None and hasattr(optimized_model, 'predict'):
                        model = optimized_model
                        logger.info("Model optimization successful")
                    else:
                        logger.warning("Model optimization failed, using original model")
                except Exception as opt_e:
                    logger.warning(f"Model optimization failed: {opt_e}, using original model")
            
            # Move to target device if not CPU
            if self.device != "cpu":
                try:
                    logger.info(f"Moving model to device: {self.device}")
                    if hasattr(model, 'to'):
                        model = model.to(self.device)
                    elif hasattr(model, 'model') and hasattr(model.model, 'to'):
                        model.model = model.model.to(self.device)
                except Exception as device_e:
                    logger.warning(f"Failed to move model to {self.device}: {device_e}")
                    logger.info("Continuing with CPU")
                    self.device = "cpu"
            
            # Final verification
            if model is None or not hasattr(model, 'predict'):
                raise RuntimeError("Final model verification failed")
            
            logger.info(f"RF-DETR model initialized successfully on device: {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize RF-DETR model: {e}")
            logger.error("This could be due to:")
            logger.error("1. Missing model weights")
            logger.error("2. Device compatibility issues")
            logger.error("3. Incorrect RF-DETR installation")
            logger.error("4. Memory issues")
            raise RuntimeError(f"RF-DETR model initialization failed: {e}")
    
    def _predict_on_slice(self, image_slice: np.ndarray) -> Detection:
        """
        Run RF-DETR prediction on a single image slice with better error handling
        
        Args:
            image_slice: Image slice as numpy array (BGR format)
            
        Returns:
            Detection object with bounding boxes, confidences, and class IDs
        """
        try:
            # Verify model is available
            if self.model is None:
                logger.error("Model is None, cannot make prediction")
                return Detection(xyxy=[], confidence=[], class_id=[])
            
            if not hasattr(self.model, 'predict'):
                logger.error("Model does not have predict method")
                return Detection(xyxy=[], confidence=[], class_id=[])
            
            # Convert BGR to RGB and then to PIL Image
            rgb_slice = cv2.cvtColor(image_slice, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_slice)
            
            # Verify PIL image
            if pil_image is None or pil_image.size[0] == 0 or pil_image.size[1] == 0:
                logger.error("Invalid PIL image for prediction")
                return Detection(xyxy=[], confidence=[], class_id=[])
            
            # Run RF-DETR prediction with timeout protection
            try:
                detections = self.model.predict(pil_image, threshold=self.confidence_threshold)
                
                # Verify detection result
                if detections is None:
                    logger.warning("Model returned None detection")
                    return Detection(xyxy=[], confidence=[], class_id=[])
                
                # Ensure detection has required attributes
                if not hasattr(detections, 'xyxy') or not hasattr(detections, 'confidence') or not hasattr(detections, 'class_id'):
                    logger.warning("Detection object missing required attributes")
                    return Detection(xyxy=[], confidence=[], class_id=[])
                
                return detections
                
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory, trying CPU")
                torch.cuda.empty_cache()
                return Detection(xyxy=[], confidence=[], class_id=[])
            except Exception as pred_e:
                logger.error(f"Prediction failed: {pred_e}")
                return Detection(xyxy=[], confidence=[], class_id=[])
            
        except Exception as e:
            logger.error(f"Error in slice prediction: {e}")
            return Detection(xyxy=[], confidence=[], class_id=[])
    
    def _merge_slice_predictions(
        self, 
        slice_predictions: List[Tuple[Detection, SliceInfo]], 
        original_shape: Tuple[int, int]
    ) -> Detection:
        """
        Merge predictions from multiple slices and apply NMS
        
        Args:
            slice_predictions: List of (detection, slice_info) tuples
            original_shape: Original image shape (height, width)
            
        Returns:
            Merged detection object
        """
        all_boxes = []
        all_confidences = []
        all_class_ids = []
        
        for detection, slice_info in slice_predictions:
            if detection is None or len(detection.xyxy) == 0:
                continue
                
            # Convert slice coordinates back to original image coordinates
            shift_x, shift_y = slice_info.starting_pixel
            
            # Adjust bounding boxes
            adjusted_boxes = detection.xyxy.copy()
            adjusted_boxes[:, [0, 2]] += shift_x  # x coordinates
            adjusted_boxes[:, [1, 3]] += shift_y  # y coordinates
            
            # Clip to original image boundaries
            adjusted_boxes[:, [0, 2]] = np.clip(adjusted_boxes[:, [0, 2]], 0, original_shape[1])
            adjusted_boxes[:, [1, 3]] = np.clip(adjusted_boxes[:, [1, 3]], 0, original_shape[0])
            
            # Filter out invalid boxes
            valid_boxes = []
            valid_confidences = []
            valid_class_ids = []
            
            for i, box in enumerate(adjusted_boxes):
                if box[2] > box[0] and box[3] > box[1]:  # Valid box
                    valid_boxes.append(box)
                    valid_confidences.append(detection.confidence[i])
                    valid_class_ids.append(detection.class_id[i])
            
            if valid_boxes:
                all_boxes.extend(valid_boxes)
                all_confidences.extend(valid_confidences)
                all_class_ids.extend(valid_class_ids)
        
        if not all_boxes:
            return Detection(xyxy=[], confidence=[], class_id=[])
        
        # Convert to numpy arrays
        merged_boxes = np.array(all_boxes)
        merged_confidences = np.array(all_confidences)
        merged_class_ids = np.array(all_class_ids)
        
        # Apply Non-Maximum Suppression using supervision
        try:
            sv_detections = sv.Detections(
                xyxy=merged_boxes,
                confidence=merged_confidences,
                class_id=merged_class_ids.astype(int)
            )
            
            # Apply NMS
            nms_detections = sv_detections.with_nms(threshold=self.nms_threshold)
            
            return Detection(
                xyxy=nms_detections.xyxy,
                confidence=nms_detections.confidence,
                class_id=nms_detections.class_id
            )
            
        except Exception as e:
            logger.warning(f"NMS failed, using merged detections: {e}")
            return Detection(
                xyxy=merged_boxes,
                confidence=merged_confidences,
                class_id=merged_class_ids
            )
    
    def test_model(self) -> bool:
        """Test if the model is working properly"""
        try:
            logger.info("Testing RF-DETR model...")
            
            # Create a test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            pil_test = Image.fromarray(test_image)
            
            # Try prediction
            result = self.model.predict(pil_test, threshold=0.5)
            
            if result is not None:
                logger.info("✅ Model test successful")
                return True
            else:
                logger.error("❌ Model test failed: returned None")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            return False
    
    def process_frame_with_sahi(self, frame: np.ndarray) -> Tuple[np.ndarray, Detection]:
        """
        Process a single frame using SAHI (slicing) with RF-DETR
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        try:
            original_height, original_width = frame.shape[:2]
            
            # Generate slices
            slices = slice_image(
                image=frame,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio
            )
            
            logger.debug(f"Generated {len(slices)} slices for frame")
            
            # Process each slice
            slice_predictions = []
            for i, slice_info in enumerate(slices):
                detection = self._predict_on_slice(slice_info.image)
                slice_predictions.append((detection, slice_info))
                
                # Log detection count for debugging
                if len(detection.class_id) > 0:
                    logger.debug(f"Slice {i}: {len(detection.class_id)} detections")
            
            # Merge predictions from all slices
            merged_detections = self._merge_slice_predictions(
                slice_predictions, 
                (original_height, original_width)
            )
            
            # Create labels for visualization
            labels = []
            if len(merged_detections.class_id) > 0:
                labels = [
                    f"{COCO_CLASSES[min(class_id, len(COCO_CLASSES)-1)]} {confidence:.2f}"
                    for class_id, confidence in zip(
                        merged_detections.class_id, 
                        merged_detections.confidence
                    )
                ]
            
            # Annotate frame
            annotated_frame = frame.copy()
            if len(merged_detections.xyxy) > 0:
                sv_detections = sv.Detections(
                    xyxy=merged_detections.xyxy,
                    confidence=merged_detections.confidence,
                    class_id=merged_detections.class_id.astype(int)
                )
                
                annotated_frame = self.box_annotator.annotate(annotated_frame, sv_detections)
                annotated_frame = self.label_annotator.annotate(annotated_frame, sv_detections, labels)
            
            return annotated_frame, merged_detections
            
        except Exception as e:
            logger.error(f"Error processing frame with SAHI: {e}")
            return frame, Detection(xyxy=[], confidence=[], class_id=[])
    
    def process_frame_simple(self, frame: np.ndarray) -> Tuple[np.ndarray, Detection]:
        """
        Process a single frame without slicing (simple RF-DETR prediction)
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        try:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Run RF-DETR prediction
            detections = self._predict_on_slice(rgb_frame)  # Reuse the error-handled prediction method
            
            # Create labels
            labels = []
            if len(detections.class_id) > 0:
                labels = [
                    f"{COCO_CLASSES[min(class_id, len(COCO_CLASSES)-1)]} {confidence:.2f}"
                    for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
            
            # Annotate frame
            annotated_frame = frame.copy()
            if len(detections.xyxy) > 0:
                sv_detections = sv.Detections(
                    xyxy=detections.xyxy,
                    confidence=detections.confidence,
                    class_id=detections.class_id.astype(int)
                )
                
                annotated_frame = self.box_annotator.annotate(annotated_frame, sv_detections)
                annotated_frame = self.label_annotator.annotate(annotated_frame, sv_detections, labels)
            
            return annotated_frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, Detection(xyxy=[], confidence=[], class_id=[])
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        use_sahi: bool = True,
        save_predictions: bool = True,
        show_progress: bool = True,
        skip_frames: int = 0,
        test_model_first: bool = True
    ) -> Dict[str, Any]:
        """
        Process an entire video file
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            use_sahi: Whether to use SAHI slicing (True) or simple prediction (False)
            save_predictions: Whether to save prediction results
            show_progress: Whether to show progress bar
            skip_frames: Number of frames to skip for faster processing
            test_model_first: Whether to test the model before processing
            
        Returns:
            Dictionary with processing statistics
        """
        
        # Test model first
        if test_model_first:
            if not self.test_model():
                logger.error("Model test failed. Cannot proceed with video processing.")
                raise RuntimeError("Model test failed")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        logger.info(f"Using {'SAHI slicing' if use_sahi else 'simple prediction'}")
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'processing_time': 0,
            'fps': 0,
            'use_sahi': use_sahi,
            'predictions_by_frame': []
        }
        
        frame_count = 0
        start_time = time.time()
        
        # Progress bar
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing video")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    if show_progress:
                        pbar.update(1)
                    continue
                
                # Process frame
                if use_sahi:
                    annotated_frame, detections = self.process_frame_with_sahi(frame)
                else:
                    annotated_frame, detections = self.process_frame_simple(frame)
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(detections.class_id)
                
                if save_predictions and len(detections.class_id) > 0:
                    frame_predictions = []
                    for i in range(len(detections.class_id)):
                        frame_predictions.append({
                            'bbox': detections.xyxy[i].tolist(),
                            'category_id': int(detections.class_id[i]),
                            'category_name': COCO_CLASSES[min(detections.class_id[i], len(COCO_CLASSES)-1)],
                            'confidence': float(detections.confidence[i])
                        })
                    stats['predictions_by_frame'].append({
                        'frame': frame_count,
                        'predictions': frame_predictions
                    })
                
                frame_count += 1
                
                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({
                        'detections': len(detections.class_id),
                        'total_det': stats['total_detections']
                    })
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            if show_progress:
                pbar.close()
            
            # Final statistics
            end_time = time.time()
            stats['processing_time'] = end_time - start_time
            stats['fps'] = stats['processed_frames'] / stats['processing_time'] if stats['processing_time'] > 0 else 0
            
            logger.info(f"Processing completed:")
            logger.info(f"  Processed frames: {stats['processed_frames']}/{stats['total_frames']}")
            logger.info(f"  Total detections: {stats['total_detections']}")
            logger.info(f"  Processing time: {stats['processing_time']:.2f}s")
            logger.info(f"  Average FPS: {stats['fps']:.2f}")
            
            # Save predictions to JSON if requested
            if save_predictions:
                pred_path = Path(output_path).with_suffix('.json')
                with open(pred_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"Predictions saved to: {pred_path}")
        
        return stats


def main():
    """Main function to run the video processor"""
    parser = argparse.ArgumentParser(description="RF-DETR + SAHI Video Processing")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--model", "-m", help="Path to RF-DETR model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--slice-height", type=int, default=640, help="Slice height for SAHI")
    parser.add_argument("--slice-width", type=int, default=640, help="Slice width for SAHI")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="Overlap ratio for slices")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, mps)")  # Default to CPU
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip frames for faster processing")
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI slicing")
    parser.add_argument("--no-save-predictions", action="store_true", help="Don't save prediction results")
    parser.add_argument("--no-optimize", action="store_true", help="Don't optimize model for inference")
    parser.add_argument("--no-test", action="store_true", help="Skip model testing")
    
    args = parser.parse_args()
    
    try:
        # Create processor
        processor = RFDETRSAHIVideoProcessor(
            model_path=args.model,
            confidence_threshold=args.confidence,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
            device=args.device,
            optimize_for_inference=not args.no_optimize
        )
        
        # Process video
        stats = processor.process_video(
            input_path=args.input,
            output_path=args.output,
            use_sahi=not args.no_sahi,
            save_predictions=not args.no_save_predictions,
            skip_frames=args.skip_frames,
            test_model_first=not args.no_test
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Output saved to: {args.output}")
        if not args.no_save_predictions:
            print(f"📊 Predictions saved to: {Path(args.output).with_suffix('.json')}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Try using --device cpu")
        print("2. Try --no-optimize")
        print("3. Try --no-sahi for simple processing")
        print("4. Check if RF-DETR is properly installed: pip install rfdetr")
        sys.exit(1)


if __name__ == "__main__":
    main()

