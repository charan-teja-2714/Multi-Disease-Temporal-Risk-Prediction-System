import json
import os
from typing import Dict, Optional
import requests

class SimpleRAGExtractor:
    """Simple RAG extractor using direct Groq API calls"""
    
    def __init__(self, groq_api_key: str):
        self.api_key = groq_api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def extract_medical_data(self, medical_text: str) -> Dict[str, Optional[float]]:
        """Extract medical data using Groq API"""
        
        prompt = f"""Extract ONLY the following medical values from this report. Return STRICT JSON format with numeric values only. If a value is not found, use null.

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
- Extract exact numeric values only

JSON:"""

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return self._parse_response(content)
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return self._empty_result()
                
        except Exception as e:
            print(f"RAG extraction error: {e}")
            return self._empty_result()
    
    def _parse_response(self, text: str) -> Dict[str, Optional[float]]:
        """Parse LLM response to extract medical values"""
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
            print(f"Raw response: {text[:200]}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Optional[float]]:
        """Return empty result structure"""
        return {
            'glucose': None, 'hba1c': None, 'creatinine': None, 'bun': None,
            'systolic_bp': None, 'diastolic_bp': None, 'cholesterol': None,
            'hdl': None, 'ldl': None, 'triglycerides': None, 'bmi': None
        }

def create_simple_rag_extractor() -> Optional[SimpleRAGExtractor]:
    """Create simple RAG extractor"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Warning: GROQ_API_KEY not found")
        return None
    
    try:
        return SimpleRAGExtractor(groq_api_key)
    except Exception as e:
        print(f"Failed to initialize simple RAG extractor: {e}")
        return None
