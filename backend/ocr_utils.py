import io
from typing import Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pdfplumber not available - PDF processing disabled")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract/PIL not available - image OCR disabled")

class OCRProcessor:
    """Handle text extraction from PDFs and images"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber"""
        if not PDF_AVAILABLE:
            return ""
            
        try:
            text_content = []
            
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            return "\n".join(text_content)
            
        except Exception as e:
            print(f"PDF text extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_image(file_content: bytes) -> str:
        """Extract text from image using pytesseract OCR"""
        if not OCR_AVAILABLE:
            return ""
            
        try:
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            return text.strip()
            
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""
    
    @staticmethod
    def is_pdf_scanned(file_content: bytes) -> bool:
        """Check if PDF is scanned (no extractable text)"""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages[:3]:  # Check first 3 pages
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        return False
            return True
        except:
            return True
    
    @staticmethod
    def extract_text_from_file(file_content: bytes, filename: str) -> str:
        """Extract text from file based on type"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            if not PDF_AVAILABLE:
                raise ValueError("PDF processing not available - install pdfplumber")
                
            # Try text extraction first
            text = OCRProcessor.extract_text_from_pdf(file_content)
            
            # If no text or very little text, treat as scanned PDF
            if not text or len(text.strip()) < 50:
                if not OCR_AVAILABLE:
                    raise ValueError("OCR not available for scanned PDF - install pytesseract and PIL")
                print("PDF appears to be scanned, using OCR")
                return OCRProcessor.extract_text_from_image(file_content)
            
            return text
            
        elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
            if not OCR_AVAILABLE:
                raise ValueError("Image OCR not available - install pytesseract and PIL")
            return OCRProcessor.extract_text_from_image(file_content)
        
        else:
            raise ValueError(f"Unsupported file type: {filename}")

def extract_medical_text(file_content: bytes, filename: str) -> Optional[str]:
    """Main function to extract text from medical reports"""
    try:
        processor = OCRProcessor()
        text = processor.extract_text_from_file(file_content, filename)
        
        if not text or len(text.strip()) < 10:
            print("Warning: Very little text extracted from file")
            return None
        
        return text.strip()
        
    except Exception as e:
        print(f"Text extraction failed: {e}")
        return None