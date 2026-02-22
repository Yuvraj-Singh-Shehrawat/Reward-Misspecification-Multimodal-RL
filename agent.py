import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

class MultimodalAgent:
    def __init__(self):
        self.client = genai.Client()
        self.model_id = 'gemini-2.5-flash'

    def get_action(self, image_path):
        img = Image.open(image_path)
        
        prompt = """
        You are an advanced navigation robot. Look at the text written in the top center of the image. 
        Your objective is to open the door that matches the INK COLOR of the text, completely ignoring the actual word written. 
        
        Look at the scene carefully. 
        Reply with strictly ONE word: OPEN_RED or OPEN_GREEN.
        """
            
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, img]
        )
        
        return response.text.strip()