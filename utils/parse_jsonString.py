import json
import re

def parse_probabilities(text: str, num_options: int = 4) -> dict:
    """
    从字符串中解析选项概率，支持字符串形式的概率值
    
    Args:
        text: JSON格式的字符串，包含选项概率
        num_options: 选项数量，如果为None则自动检测
        
    Returns:
        dict: 包含选项概率的字典，如果解析失败返回None
    """
    
    text = text.replace("'", '"')
    text = text.replace('```json', '').replace('```', '')
    text = text.strip()
    valid_options = [chr(65 + i) for i in range(num_options)]
    
    try:
        
        probabilities = json.loads(text)
        
        if isinstance(probabilities, str):
            probabilities = json.loads(probabilities)
        if not isinstance(probabilities, dict):
            raise json.JSONDecodeError("probabilities is not a dict", text, 0)
        
        
        if not all(key in valid_options for key in probabilities.keys()):
            print(f"option format error, expected options: {valid_options}, actual options: {list(probabilities.keys())}")
            return None
        
        
        converted_probs = {}
        for key, value in probabilities.items():
            try:
                
                if isinstance(value, str):
                    
                    value = value.strip('"\'').strip()
                    
                    converted_probs[key] = float(value)
                else:
                    converted_probs[key] = float(value)
            except (ValueError, TypeError):
                print(f"cannot convert value to float: {key}={value}")
                return None
            
        
        if num_options:
            missing_options = set(valid_options) - set(converted_probs.keys())
            if missing_options:
                print(f"option number mismatch, fill in missing options: {missing_options}")
                for option in missing_options:
                    converted_probs[option] = 0.0  # set the probability of missing options to 0
        
        return converted_probs
        
    except json.JSONDecodeError:
        
        pattern = r'"([A-Z])"\s*:\s*"?(\d+(?:\.\d+)?)"?'
        matches = re.findall(pattern, text)
        
        if not matches:
            print("cannot parse answer format")
            return None
            
        
        probabilities = {}
        for option, value in matches:
            try:
                probabilities[option] = float(value)
            except ValueError:
                print(f"cannot convert value to float: {option}={value}")
                return None
        
        
        if num_options:
            missing_options = set(valid_options) - set(probabilities.keys())
            if missing_options:
                print(f"option number mismatch, fill in missing options: {missing_options}")
                for option in missing_options:
                    probabilities[option] = 0.0  # set the probability of missing options to 0
        
        return probabilities
    
def parse_json_string(text: str, expected_keys: list = None, validate_types: dict = None) -> dict:
    """
    parse any JSON format data from string
    
    Args:
        text: string containing JSON data
        expected_keys: optional, list of keys expected in JSON
        validate_types: optional, dictionary of key-value pairs, specifying the type of the value for a specific key
        
    Returns:
        dict: parsed JSON dictionary, return None if parsing fails
    """
    
    if '```' in text:
        
        text = re.sub(r'```(?:json|python|javascript)?', '', text)
        text = text.replace('```', '')
    
    text = text.strip()
    
    try:
        
        parsed_data = json.loads(text)
        
        
        if not isinstance(parsed_data, dict):
            print(f"warning: parsed result is not a dictionary, but {type(parsed_data).__name__}")
            return {"value": parsed_data}
        
        
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in parsed_data]
            if missing_keys:
                print(f"missing expected keys: {missing_keys}")
                return None
        
        
        if validate_types:
            for key, expected_type in validate_types.items():
                if key in parsed_data:
                    if not isinstance(parsed_data[key], expected_type):
                        actual_type = type(parsed_data[key]).__name__
                        expected_type_name = expected_type.__name__
                        print(f"key '{key}' value type error, expected {expected_type_name}, actual {actual_type}")
                        return None
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        
        
        try:
            
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except:
            pass
            
        try:
            
            fixed_text = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', text)
            return json.loads(fixed_text)
        except:
            pass
        
        
        option_pattern = r'"([A-Za-z0-9_]+)"\s*:\s*(\d+(?:\.\d+)?)'
        matches = re.findall(option_pattern, text)
        
        if matches:
            parsed_data = {option: float(value) for option, value in matches}
            return parsed_data
            
        return None
    
    except Exception as e:
        print(f"error parsing JSON: {str(e)}")
        return None