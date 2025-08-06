import time
import openai
import requests
def test_model_api(model_name, api_base, api_key):
    if "CoVR" in model_name:
        print(f"✅ API test success: {model_name}")
        return True
    if "ocr" or "majority" or "few_shot" in model_name:
        model_name = model_name.replace("_ocr", "")
        model_name = model_name.replace("_majority", "")
        model_name = model_name.replace("_few_shot", "")
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    retries = 0
    while True:  # infinite loop until API is available
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print(f"✅ API test success: {model_name}")
            return True
        except (openai.APIError, openai.APIConnectionError, 
                openai.RateLimitError, requests.exceptions.RequestException) as e:
            retries += 1
            wait_time = 5  # gradually increase waiting time, but max 60 seconds
            print(f"❌ API test failed (attempt #{retries}): {model_name} - {str(e)}")
            print(f"⏳ waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ unknown error: {model_name} - {str(e)}")
            print("⏳ waiting 5 seconds before retrying...")
            time.sleep(5)