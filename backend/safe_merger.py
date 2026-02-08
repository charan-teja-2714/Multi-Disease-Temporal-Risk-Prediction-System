from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from database import HealthRecord, Patient
from datetime import datetime

class SafeMerger:
    """Handle safe merging of RAG-extracted data with existing records"""
    
    @staticmethod
    def get_latest_health_record(db: Session, patient_id: int) -> Optional[HealthRecord]:
        """Get the most recent health record for a patient"""
        return db.query(HealthRecord).filter(
            HealthRecord.patient_id == patient_id
        ).order_by(HealthRecord.visit_date.desc()).first()
    
    @staticmethod
    def merge_extracted_data(
        db: Session, 
        patient_id: int, 
        extracted_data: Dict[str, Optional[float]]
    ) -> Optional[HealthRecord]:
        """Safely merge extracted data without overwriting existing values"""
        
        # Verify patient exists
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")
        
        # Get latest record for reference
        latest_record = SafeMerger.get_latest_health_record(db, patient_id)
        
        # Prepare merged data
        merged_data = {}
        has_new_data = False
        
        field_names = [
            'glucose', 'hba1c', 'creatinine', 'bun', 'systolic_bp',
            'diastolic_bp', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'bmi'
        ]
        
        for field in field_names:
            extracted_value = extracted_data.get(field)
            
            if latest_record:
                existing_value = getattr(latest_record, field)
                
                # Keep existing value if it exists, otherwise use extracted
                if existing_value is not None:
                    merged_data[field] = existing_value
                else:
                    merged_data[field] = extracted_value
                    if extracted_value is not None:
                        has_new_data = True
            else:
                # No existing record, use extracted data
                merged_data[field] = extracted_value
                if extracted_value is not None:
                    has_new_data = True
        
        # Only create new record if we have at least one valid extracted value
        if not has_new_data:
            print("No new data to merge")
            return None
        
        # Create new health record
        new_record = HealthRecord(
            patient_id=patient_id,
            visit_date=datetime.utcnow(),
            source="rag_report",
            **merged_data
        )
        
        try:
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            
            print(f"Created new health record (ID: {new_record.id}) with RAG data")
            return new_record
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to save merged record: {e}")
    
    @staticmethod
    def validate_extracted_data(extracted_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Validate and clean extracted data"""
        validated = {}
        
        # Define reasonable ranges for validation
        ranges = {
            'glucose': (50, 500),      # mg/dL
            'hba1c': (4.0, 15.0),     # %
            'creatinine': (0.5, 10.0), # mg/dL
            'bun': (5, 100),          # mg/dL
            'systolic_bp': (70, 250),  # mmHg
            'diastolic_bp': (40, 150), # mmHg
            'cholesterol': (100, 500), # mg/dL
            'hdl': (20, 150),         # mg/dL
            'ldl': (50, 400),         # mg/dL
            'triglycerides': (30, 1000), # mg/dL
            'bmi': (10, 60)           # kg/m²
        }
        
        for field, value in extracted_data.items():
            if value is None:
                validated[field] = None
                continue
            
            # Check if value is in reasonable range
            if field in ranges:
                min_val, max_val = ranges[field]
                if min_val <= value <= max_val:
                    validated[field] = value
                else:
                    print(f"Warning: {field} value {value} outside normal range [{min_val}, {max_val}]")
                    validated[field] = None
            else:
                validated[field] = value
        
        return validated
    
    @staticmethod
    def get_merge_summary(
        extracted_data: Dict[str, Optional[float]], 
        latest_record: Optional[HealthRecord]
    ) -> Dict[str, str]:
        """Generate summary of what will be merged"""
        summary = {
            'new_fields': [],
            'existing_fields': [],
            'extracted_count': 0,
            'merged_count': 0
        }
        
        for field, value in extracted_data.items():
            if value is not None:
                summary['extracted_count'] += 1
                
                if latest_record:
                    existing_value = getattr(latest_record, field)
                    if existing_value is not None:
                        summary['existing_fields'].append(field)
                    else:
                        summary['new_fields'].append(field)
                        summary['merged_count'] += 1
                else:
                    summary['new_fields'].append(field)
                    summary['merged_count'] += 1
        
        return summary