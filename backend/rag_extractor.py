import json
import re
from typing import Dict, Optional
import os

try:
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available - RAG extraction disabled")

class MedicalDataParser:
    """Custom parser for medical data extraction"""
    
    def parse(self, text: str) -> Dict[str, Optional[float]]:
        """Parse LLM output to extract medical values"""
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            text = text.strip()
            data = json.loads(text)
            
            result = {}
            valid_fields = [
                'glucose', 'hba1c', 'creatinine', 'bun', 'systolic_bp',
                'diastolic_bp', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'bmi'
            ]
            
            for field in valid_fields:
                value = data.get(field)
                if value is not None:
                    try:
                        result[field] = float(value)
                    except (ValueError, TypeError):
                        result[field] = None
                else:
                    result[field] = None
            
            return result
            
        except Exception as e:
            print(f"Parser error: {e}")
            print(f"Raw response: {text[:200] if text else 'empty'}")
            return {field: None for field in [
                'glucose', 'hba1c', 'creatinine', 'bun', 'systolic_bp',
                'diastolic_bp', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'bmi'
            ]}

class RAGExtractor:
    """Medical report data extraction using Groq + LangChain"""
    
    def __init__(self, groq_api_key: str):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
            
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        self.parser = MedicalDataParser()
        
        self.prompt_template = """Extract ONLY the following medical values from this report. Return STRICT JSON format with numeric values only. If a value is not found, use null.

Medical Report Text:
{medical_text}

Extract these fields (use exact field names):
- glucose (blood sugar, FBS, fasting glucose) in mg/dL
- hba1c (glycated hemoglobin, A1C) in %
- creatinine in mg/dL
- bun (blood urea nitrogen) in mg/dL
- systolic_bp (systolic blood pressure) in mmHg
- diastolic_bp (diastolic blood pressure) in mmHg
- cholesterol (total cholesterol) in mg/dL
- hdl (HDL cholesterol) in mg/dL
- ldl (LDL cholesterol) in mg/dL
- triglycerides in mg/dL
- bmi (body mass index) in kg/m²

RULES:
- Return ONLY valid JSON
- Use null for missing values
- Do NOT explain or add comments
- Do NOT diagnose or interpret
- Extract exact numeric values only

JSON:"""
    
    def extract_medical_data(self, medical_text: str) -> Dict[str, Optional[float]]:
        """Extract structured medical data from raw text"""
        if not LANGCHAIN_AVAILABLE:
            return self._empty_result()
            
        try:
            prompt = self.prompt_template.format(medical_text=medical_text)
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            result = self.parser.parse(content)
            
            if not isinstance(result, dict):
                return self._empty_result()
            
            return result
            
        except Exception as e:
            print(f"RAG extraction error: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Optional[float]]:
        """Return empty result structure"""
        return {
            'glucose': None,
            'hba1c': None,
            'creatinine': None,
            'bun': None,
            'systolic_bp': None,
            'diastolic_bp': None,
            'cholesterol': None,
            'hdl': None,
            'ldl': None,
            'triglycerides': None,
            'bmi': None
        }

def create_rag_extractor() -> Optional[RAGExtractor]:
    """Factory function to create RAG extractor"""
    if not LANGCHAIN_AVAILABLE:
        print("Warning: LangChain not installed - RAG extraction unavailable")
        return None
        
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Warning: GROQ_API_KEY not found in environment")
        return None
    
    try:
        return RAGExtractor(groq_api_key)
    except Exception as e:
        print(f"Failed to initialize RAG extractor: {e}")
        return None