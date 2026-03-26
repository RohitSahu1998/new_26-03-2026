import pandas as pd
import cv2
import numpy as np
import re
from PIL import Image

def clean_text(text):
    """Normalize text for fuzzy matching by removing special characters and keeping alphanumeric + spaces"""
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).strip()

def flatten_json(y):
    """Flattens nested JSON into a single level dictionary of keys to scalar values."""
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x
            
    flatten(y)
    return out

def match_qwen_to_ocr(qwen_json, ocr_results):
    """
    Matches Qwen extracted values to PaddleOCR text blocks to recover their bounding boxes.
    """
    # Support flattened single-page testing by wrapping it inside page_1 if needed
    if isinstance(qwen_json, dict) and not any(k.startswith('page_') for k in qwen_json.keys()):
        qwen_json = {"page_1": qwen_json}

    matched_data = []

    for page_key, page_content in qwen_json.items():
        if not page_key.startswith('page_'):
            continue
            
        page_num_str = page_key.replace("page_", "")
        page_num = int(page_num_str) if page_num_str.isdigit() else 1
        
        # Filter OCR results purely for this current page
        page_ocr_results = [box for box in ocr_results if box.get('page', 1) == page_num]
        
        flattened_qwen = flatten_json(page_content)
        
        used_ocr_indices = set()
        used_ocr_ranges = {} # Tracks which character parts of a huge OCR block are "consumed" { idx: [(start1, end1), ...] }

        for key, qwen_value in flattened_qwen.items():
            if not qwen_value or str(qwen_value).strip() == "":
                continue
                
            target_clean = clean_text(qwen_value)
            if not target_clean or len(target_clean) < 2:
                continue
                
            remaining_target = target_clean
                
            for idx, ocr_box in enumerate(page_ocr_results):
                if len(remaining_target) < 2:
                    break
                    
                box_text_clean = clean_text(ocr_box['text'])
                if not box_text_clean:
                    continue
                
                # ========================
                # CASE 1: Exact matches ONLY to prevent dates/prices from incorrectly cross-matching
                # ========================
                if idx not in used_ocr_indices and box_text_clean in remaining_target:
                    matched_data.append({
                        "Page": page_num,
                        "Key": key,
                        "Qwen_Value": str(qwen_value),
                        "OCR_Matched_Text": ocr_box['text'],
                        "Confidence": ocr_box['confidence'],
                        "BBox": ocr_box['bbox']
                    })
                    used_ocr_indices.add(idx)
                    remaining_target = remaining_target.replace(box_text_clean, "", 1)
                    continue 

                # ========================
                # CASE 2: Qwen field is buried inside a massive OCR line
                # ========================
                if remaining_target in box_text_clean and len(remaining_target) >= 4:
                    import re as regex_mod
                    # Find all occurrences of the remaining_target inside the OCR box
                    start_indices = [m.start() for m in regex_mod.finditer(regex_mod.escape(remaining_target), box_text_clean)]
                    
                    valid_start = None
                    consumed_ranges = used_ocr_ranges.get(idx, [])
                    
                    # Ensure we claim an occurrence that wasn't already consumed by another Key!
                    for start_idx in start_indices:
                        end_idx = start_idx + len(remaining_target)
                        overlap = False
                        for rs, re_bound in consumed_ranges:
                            if max(start_idx, rs) < min(end_idx, re_bound):
                                overlap = True
                                break
                        if not overlap:
                            valid_start = start_idx
                            break
                            
                    if valid_start is None:
                        continue # All occurrences were already consumed!
                        
                    best_start = valid_start
                    best_end = valid_start + len(remaining_target)
                    
                    # Mark this specific chunk of the OCR box as "Consumed"
                    if idx not in used_ocr_ranges:
                        used_ocr_ranges[idx] = []
                    used_ocr_ranges[idx].append((best_start, best_end))
                    
                    # Calculate character position ratios for geometric slicing
                    start_ratio = best_start / max(1, len(box_text_clean))
                    end_ratio = best_end / max(1, len(box_text_clean))
                    
                    x1, y1 = ocr_box['bbox'][0]
                    x2, y2 = ocr_box['bbox'][1]
                    x3, y3 = ocr_box['bbox'][2]
                    x4, y4 = ocr_box['bbox'][3]
                    
                    top_width = x2 - x1
                    bottom_width = x3 - x4
                    
                    new_x1 = x1 + (top_width * start_ratio)
                    new_x2 = x1 + (top_width * end_ratio)
                    new_x4 = x4 + (bottom_width * start_ratio)
                    new_x3 = x4 + (bottom_width * end_ratio)
                    
                    sub_bbox = [
                        [new_x1, y1], [new_x2, y2], [new_x3, y3], [new_x4, y4]
                    ]
                    
                    matched_data.append({
                        "Page": page_num,
                        "Key": key,
                        "Qwen_Value": str(qwen_value),
                        "OCR_Matched_Text": f"{str(qwen_value)} (Sub)",
                        "Confidence": ocr_box['confidence'],
                        "BBox": sub_bbox
                    })
                    remaining_target = ""
                    break

    return matched_data

def highlight_matches_on_image(document_path, matched_data, output_path):
    """
    Draws seamless light green highlights across multi-page PDFs or single images.
    """
    images = []
    
    # Safely load the document (Natively supporting Multi-Page PDFs)
    if document_path.lower().endswith('.pdf'):
        from pdf2image import convert_from_path
        pil_pages = convert_from_path(document_path)
        for page in pil_pages:
            # OpenCV requires BGR formatting
            images.append(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
    else:
        img = cv2.imread(document_path)
        if img is None:
            print(f"Error: Could not read image at {document_path}.")
            return
        images.append(img)
        
    final_pil_pages = []
    
    # Process highlights iteratively for every physical page
    for i, image in enumerate(images):
        page_num = i + 1
        page_matched = [m for m in matched_data if m.get('Page', 1) == page_num]
        
        overlay = image.copy()
        
        # Group coordinates by their Qwen Key
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for m in page_matched:
            bbox = m['BBox']
            grouped[m['Key']].append(bbox)
            
            # Fill Polygon Rectangle (Light Green)
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (144, 238, 144))
            
        # Blend the green color over the image seamlessly at 50% opacity
        alpha = 0.5
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Superimpose the red key textual label above the grouping
        for key, bboxes in grouped.items():
            top_left = min([b[0] for b in bboxes], key=lambda x: (x[1], x[0]))
            x, y = int(top_left[0]), int(top_left[1])
            cv2.putText(image, str(key), (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Revert colors back to RGB to save via Pillow
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_pil_pages.append(Image.fromarray(rgb_image))
        
    # Multi-page PDF Export Generator
    if output_path.lower().endswith('.pdf') and final_pil_pages:
        final_pil_pages[0].save(
            output_path, "PDF", resolution=100.0, 
            save_all=True, append_images=final_pil_pages[1:]
        )
        print(f"✅ Multi-page Highlighted PDF successfully saved to: {output_path}")
    elif final_pil_pages:
        # If the user specifically requested JPG format, we write out the first page
        cv2.imwrite(output_path, cv2.cvtColor(np.array(final_pil_pages[0]), cv2.COLOR_RGB2BGR))
        print(f"✅ Highlighted image successfully saved to: {output_path}")

def export_to_excel(matched_data, excel_output_path="matched_results.xlsx"):
    """
    Dumps the final matched data to an Excel file.
    """
    df = pd.DataFrame(matched_data)
    df.to_excel(excel_output_path, index=False)
    print(f"✅ Extracted data successfully exported to: {excel_output_path}")

if __name__ == "__main__":
    import os
    print("Integration pipeline loaded successfully.")
    
    test_document_path = "/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_1.pdf"
    
    if os.path.exists(test_document_path):
        print(f"Running Standalone Pipeline on: {test_document_path}")
        from ocr_engine import PaddleOCREngine
        from qwen_engine import QwenExtractor
        from pdf2image import convert_from_path
        
        ocr_engine = PaddleOCREngine()
        qwen_engine = QwenExtractor()
        
        ocr_output = ocr_engine.extract_text_with_confidence(test_document_path)
        qwen_output = qwen_engine.extract_data(test_document_path)
        
        matched_results = match_qwen_to_ocr(qwen_output, ocr_output)
        
        highlight_matches_on_image(test_document_path, matched_results, "output.pdf")
        export_to_excel(matched_results, "matched_results.xlsx")
        print("\n✅ Standalone run fully complete! Check 'output.pdf' and 'matched_results.xlsx'")
    else:
        print(f"\n⚠️ Could not find test file: {test_document_path}")
