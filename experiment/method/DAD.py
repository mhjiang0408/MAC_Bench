import os
import sys
sys.path.append(os.getcwd())
from utils.llm_call import CallLLM
from utils.parse_jsonString import parse_json_string
import base64

class description_extractor():
    def __init__(self,model_name="qwen2.5-vl-7b-instruct", low_detail:bool=False, ocr:bool=False, api_base:str='http://xxx', api_key:str='sk-xxx'):
        self.llm = CallLLM(model=model_name,api_base=api_base,api_key=api_key)
        self.low_detail = low_detail
        self.ocr = ocr
    def extract_description(self, image_path:str, question:str, options:str):

        

        system_prompt = "I will provide you with a scientific journal cover image. Please explain in detail the visual elements of the image. Faithfully describe the information in the image without making any speculation or judgment. Your answer should begin with “The image shows”. Answer the more the better."
        user_prompt = f"I'm blind. Here is the image. Please think step-by-step and describe the image in detail and present your answer in the form of a Pseudo-CoT. After you describe the image, please think the question step by step but do not give your final answer: {question+options}"
        if self.ocr:
            image_path = image_path.replace('Cover','OCRed_Cover')
        with open(image_path, 'rb') as image_file:
            
            base64_data = base64.b64encode(image_file.read())
            
            image_base64 = base64_data.decode('utf-8')
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "low"
                    }
                },
            ]
        }
        ]
        response, total_tokens = self.llm.post_request(message)
        return response, total_tokens
    
    def extract_descriptions(self, image_paths:list[str], question:str, story:str):
        system_prompt = "You are an excellent scientific image reader. I will provide you with 4 scientific journal cover images in the order of A, B, C, and D. Please explain in detail the visual elements of each image and analyze which image best describes the story. Faithfully describe the information in the image without making any judgment. Your answer should begin with “The image A shows”, “The image B shows”, “The image C shows”, and “The image D shows”. The Answer the more the better."
        user_prompt = f"I'm blind. Here are the images. Please think step-by-step and describe the images in detail and present your answer in the form of a Pseudo-CoT. After you describe the images, please think the question step by step but do not give your final answer: {question}. The story is {story}. The following images provided to you are, in order, A, B, C, and D. DO NOT GIVE YOUR FINAL ANSWER."
        images_base64 = []
        for image_path in image_paths:
            if self.ocr:
                image_path = image_path.replace('Cover','OCRed_Cover')
            with open(image_path, 'rb') as image_file:
                base64_data = base64.b64encode(image_file.read())
                image_base64 = base64_data.decode('utf-8')
            images_base64.append(image_base64)
        if self.low_detail:
            detail = 'low'
        else:
            detail = 'low'
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{images_base64[0]}",
                        "detail": detail
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{images_base64[1]}",
                        "detail": detail
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{images_base64[2]}",
                        "detail": detail
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{images_base64[3]}",
                        "detail": detail
                    }
                }
            ]}
        ]
        response, total_tokens = self.llm.post_request(message)
        return response, total_tokens

class reasoning_thinker():
    def __init__(self,model_name="qwq-32b",api_base:str='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key:str='sk-xxx'):
        self.llm = CallLLM(model=model_name,api_base=api_base,api_key=api_key)

    def reasoning_thinking(self, description:str, options:str):
        system_prompt = "Following the above image description, think critically step by step and predict the probability that you would choose each option. Answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"

        user_prompt = f"The image description is: {description}. The answer in image description may be wrong. And you have the following options: {options}. My question is: Which option best describes the cover image? Please predict the probability that you would choose each option."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, total_tokens, reasoning_content = self.llm.post_reasoning_request(message)
        return response, total_tokens, reasoning_content
    
    def reasoning_thinking_text2image(self,descriptions:str, story:str):
        system_prompt = "Following the image descriptions, think critically step by step and predict the probability that you would choose each option. Answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"
        user_prompt = f"The options are the images. The image descriptions are: {descriptions}. The answer in image descriptions may be wrong. My question is: Which image best describes the cover story? The story is: {story}. Please predict the probability that you would choose each option."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, total_tokens, reasoning_content = self.llm.post_reasoning_request(message)
        return response, total_tokens, reasoning_content

class story_extractor():
    def __init__(self,model_name="qwq-32b"):
        self.llm = CallLLM(model=model_name)

    def extract_stories(self, stories:str):

        system_prompt = "You are an excellent storyteller. I will provide you with four scientific stories in options from a multiple-choice question. You need to retell these four stories for elementary school students, so please distill the Scientific Concepts and focus on the parts of the stories that can be illustrated. DONOT USE rhetorical devices. # Response Format\n ```json\n { \"A\": Your new story for option A, \"B\": Your new story for option B, \"C\": Your new story for option C, \"D\": Your new story for option D }\n```"
        user_prompt = f"Here is the options: {stories}. "
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, total_tokens,_ = self.llm.post_reasoning_request(message)
        return response, total_tokens
    
    def extract_story(self,story):
        system_prompt = "You are an excellent storyteller. I will provide you with a scientific story from a multiple-choice question. You need to retell this story for elementary school students, so please distill the scientific concepts and focus on the parts of the story that can be illustrated. # Response Format\n ```json\n { \"Story\": Your new story }\n```"
        user_prompt = f"Here is the story: {story}. "
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, total_tokens,_ = self.llm.post_reasoning_request(message)
        return response, total_tokens

class child_understander():
    def __init__(self,model_name="qwen2.5-vl-7b-instruct"):
        self.llm = CallLLM(model=model_name)

    def child_understanding(self, question, options, given_info, type:str='image2text'):
        if type == 'image2text':
        
            system_prompt = "# Requirement\n You are an excellent scientific image reader. You need to analyze the provided image and choose the most appropriate option based on your understanding. ONLY based on the image and the options provided above, predict the probability that you would choose each option and answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"
            user_prompt = f"I want to ask you the following question: {question} And you have the following options: {options}. You need to predict the probability that you would choose each option based on the image."
            with open(given_info, 'rb') as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "auto"
                        }
                    }
                ]}
            ]
            response, total_tokens = self.llm.post_request(messages)
            return response, total_tokens
        elif type == 'text2text':
            base64_list = []
            for option in options:
                try:
                    with open(option, 'rb') as image_file:
                        
                        base64_data = base64.b64encode(image_file.read())
                        
                        image_base64 = base64_data.decode('utf-8')
                        base64_list.append(image_base64)
                except Exception as e:
                    print(f"Error: {e}")
                    return None
            system_prompt = "# Requirement\n You are an excellent scientific image reader. You need to analyze the provided cover story and choose the most appropriate option images based on your understanding. ONLY based on the cover story and the options provided above, predict the probability that you would choose each option and answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"
            user_prompt = f"I want to ask you the following question: {question} And you have the cover story {given_info}. You need to predict the probability that you would choose each option based on the cover story. The following images provided to you are, in order, A, B, C, and D."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[0]}",
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[1]}",
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[2]}",
                            "detail": "auto"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_list[3]}",
                            "detail": "auto"
                        }
                    }
                ]}
            ]
            response, total_tokens = self.llm.post_request(messages)
            return response, total_tokens
            

class CoVRr0():
    def __init__(self,mllm_model="qwen2.5-vl-7b-instruct", reasoning_model="qwq-32b"):
        self.child_agent = child_understander(mllm_model)
        self.story_agent = story_extractor(reasoning_model)
    def post_request(self,question, given_info, options:list,messages:list,type:str='image2text'):
        """
        options shape: [('option_0', 'option_content_0'), ('option_1', 'option_content_1'), ('option_2', 'option_content_2'), ('option_3', 'option_content_3')]
        messages is for interface consistency
        """
        if type == 'image2text':
            
            options_content = []
            for option_id,option_content in options:
                options_content.append(option_content)
            options_content = '\n'.join(options_content)
            new_options,tokens1 = self.story_agent.extract_stories(options_content)
            parsed_options = parse_json_string(new_options)
            # print(f"parsed_options: {parsed_options}")
            new_options = '\n'.join([f'{opt_id}: {parsed_options[opt_id]}' for opt_id in ['A','B','C','D']])
            response, total_tokens = self.child_agent.child_understanding(question, new_options, given_info, type=type)
            return response, total_tokens+tokens1, new_options
        elif type == 'text2text':
            
            descriptions = []
            total_tokens = 0
            # 
            new_story, tokens1 = self.story_agent.extract_story(given_info)
            parsed_story = parse_json_string(new_story,['Story'])
            new_options = [option_content for option_id, option_content in options]
            response, total_tokens2 = self.child_agent.child_understanding(question, new_options, parsed_story['Story'], type=type)
            return response, total_tokens2+tokens1, new_options

class CoVRr3():
    def __init__(self,mllm_model="qwen2.5-vl-7b-instruct", reasoning_model="qwq-32b", low_detail:bool=False,ocr:bool=False,mllm_api_base:str='http://xxx',mllm_api_key:str='sk-xxx',reasoning_api_base:str='http://xxx',reasoning_api_key:str='sk-xxx'):
        self.description_agent = description_extractor(mllm_model, low_detail,ocr,mllm_api_base,mllm_api_key)
        self.reasoning_agent = reasoning_thinker(reasoning_model,reasoning_api_base,reasoning_api_key)
    def post_request(self,question, given_info, options:list,messages:list,type:str='image2text'):
        if type == 'image2text':
            options = '\n'.join([f'{opt_id}: {opt_content}' for opt_id,opt_content in options])
            description, tokens1 = self.description_agent.extract_description(given_info, question, options)
            response, total_tokens2,reasoning_content = self.reasoning_agent.reasoning_thinking(description, options)
            return response, total_tokens2+tokens1, f'{{"description":"{description}","options":"{options}","reasoning_content":"{reasoning_content}"}}'
        elif type == 'text2image':
            new_options = [path for opt_id,path in options]
            formatted_options = '\n'.join([f'{opt_id}: {opt_content}' for opt_id,opt_content in options])
            descriptions, tokens1 = self.description_agent.extract_descriptions(new_options, question, given_info)
            response, total_tokens2,reasoning_content = self.reasoning_agent.reasoning_thinking_text2image(descriptions, given_info)
            return response, total_tokens2+tokens1, f'{{"descriptions":"{descriptions}","options":"{formatted_options}","reasoning_content":"{reasoning_content}"}}'
        else:
            raise ValueError(f"Invalid type: {type}")
    
    def post_existing_request(self,question, description:str, options:list,messages:list,type:str='image2text'):
        if type == 'image2text':
            response, total_tokens2,reasoning_content = self.reasoning_agent.reasoning_thinking(description, options)
            return response, total_tokens2, f'{{"description":"{description}","options":"{options}","reasoning_content":"{reasoning_content}"}}'
        else:
            raise ValueError(f"Invalid type: {type}")



if __name__ == "__main__": 
    covr = CoVRr3()
    images_list = [['A','MAC_Bench/ACS/Cover/ACS Applied Bio Materials/2018_1.png'], ['B','MAC_Bench/ACS/Cover/ACS Applied Bio Materials/2018_2.png'], ['C','MAC_Bench/ACS/Cover/ACS Applied Bio Materials/2018_3.png'], ['D','MAC_Bench/ACS/Cover/ACS Applied Bio Materials/2018_4.png']]
    story = 'The cover image depicts a hydrogel for wound healing containing silver nanoparticles produced by gamma irradiation; these nanoparticles act as a shield protecting from any bacteria, while the hydrogel provides a moisture environment for the wound to recover. In one step using gamma irradiation, Ag+ are reduced leading to stabilization of nanosilver but also have hydrogel formation with terminal sterilization. Because of the potential effect of silver nanoparticles crosslinked in between the hydrogel, it leads to a fast wound healing, which makes it possible to identify its mechanisms with cell regeneration.'
    response, total_tokens, reasoning_content = covr.post_request('Which option best describes the story?', story, images_list, ['lalala','lalala','lalala','lalala'],type='text2image')
    print(response)
    print(total_tokens)
    print(reasoning_content)
