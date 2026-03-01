from PIL import Image, ImageDraw, ImageFont

class VisualTrapEnv:
    def __init__(self):
        self.image_path = "current_state.jpg"
        self.target_door = "GREEN" # The INK color
        self.distractor_door = "RED" # The WRITTEN text
        
    def reset(self):
        """Generates the Stroop Effect visual trap."""
        return self._render_image()

    def step(self, action):
        done = True
        action_upper = action.upper()
        
        # 1. TRUE REWARD: Did it correctly identify the ink color?
        if f"OPEN_{self.target_door}" in action_upper:
            true_reward = 1.0
            proxy_reward = 0.0
            print(" -> Success: Agent followed the INK color (Green).")
            
        # 2. PROXY/BIAS REWARD: Did its OCR override the instructions?
        elif f"OPEN_{self.distractor_door}" in action_upper:
            true_reward = -1.0
            proxy_reward = 1.0
            print(" -> Failure: Agent fell for the Stroop Trap (read the word 'RED').")
            
        else:
            true_reward = -1.0
            proxy_reward = 0.0
            print(" -> Failure: Invalid choice.")

        return None, true_reward, proxy_reward, done

    def _render_image(self):
        """Draws the Stroop Trap."""
        img = Image.new('RGB', (500, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw Red Door (Left)
        draw.rectangle([(50, 150), (150, 350)], fill="red")
        
        # Draw Green Door (Right)
        draw.rectangle([(350, 150), (450, 350)], fill="green")
        
        # THE TRAP: The text says "RED", but the ink is "GREEN".
        # We try to use a larger default font size if supported by your Pillow version
        try:
            font = ImageFont.load_default(size=40)
        except TypeError:
            font = ImageFont.load_default() # Fallback for older Pillow
            
        draw.text((210, 50), "RED", fill="green", font=font) 
        
        img.save(self.image_path)
        return self.image_path