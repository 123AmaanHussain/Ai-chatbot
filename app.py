# # OpenAI API key: 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
# #import openai
# from flask import Flask, render_template, request,jsonify
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Initialize the Flask app
# app = Flask(__name__)

# # Set your OpenAI API key
# #openai.api_key = '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'

# # @app.route("/")
# # def home():
# #     return render_template("index.html")

# # # @app.route("/get")
# # # def get_bot_response():
# # #     user_text = request.args.get('msg')
    
# # #     # Generate response using OpenAI API with the older version syntax
# # #     response = openai.Completion.create(
# # #         engine="text-davinci-003",  # You can also use "text-curie-001" or "text-babbage-001"
# # #         prompt=f"User: {user_text}\nBot:",
# # #         max_tokens=150,
# # #         temperature=0.7,
# # #         n=1,
# # #         stop=["User:", "Bot:"]
# # #     )
    
# # #     bot_response = response['choices'][0]['text'].strip()
# # #     return bot_response

# # @app.route("/get")
# # def get_bot_response():
# #     user_text = request.args.get('msg')
    
# #     # Generate response using OpenAI API with gpt-3.5-turbo model
# #     response = openai.ChatCompletion.create(
# #         model="gpt-3.5-turbo",
# #         messages=[
# #             {"role": "system", "content": "You are a helpful assistant."},
# #             {"role": "user", "content": user_text}
# #         ],
# #         max_tokens=150,
# #         temperature=0.7
# #     )
    
# #     bot_response = response['choices'][0]['message']['content']
# #     return bot_response


# # if __name__ == "__main__":
# #     app.run(debug=True)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get", methods=["GET","Post"])
# def chat():
#     # user_text = request.args.get('msg')
    
#     # # Generate response using OpenAI API with the modern syntax
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-3.5-turbo",  # or "gpt-4" if you have access
#     #     messages=[
#     #         {"role": "system", "content": "You are a helpful assistant."},
#     #         {"role": "user", "content": user_text}
#     #     ],
#     #     max_tokens=150,
#     #     temperature=0.7
#     # )
    
#     # bot_response = response['choices'][0]['message']['content']
#     # return bot_response
#     msg = request.form['msg']
#     input = msg
#     return get_Chat_response(input)

# def get_Chat_response(text):
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

#     # Let's chat for 5 lines
#     for step in range(5):
#         # encode the new user input, add the eos_token and return a tensor in Pytorch
#         new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

#         # append the new user input tokens to the chat history
#         bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

#         # generated a response while limiting the total chat history to 1000 tokens, 
#         chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

#         # pretty print last ouput tokens from bot
#         return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


# if __name__ == "__main__":
#     app.run(debug=True)

import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

app = Flask(__name__)

# Load the conversational model from Hugging Face
model_name = "microsoft/DialoGPT-medium"  # You can replace with a more specialized model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the response context and prompt instructions
RESPONSE_TEMPLATES = {
    "anti_doping": [
        "As your anti-doping advisor, it's important to avoid substances on the WADA (World Anti-Doping Agency) prohibited list. Always consult the latest resources and check your supplements carefully. If you're unsure about an ingredient, consider talking to an expert.",
        "Avoid supplements with unfamiliar ingredients, as they might contain banned substances. Staying informed and regularly checking the WADA website can help you stay compliant and protect your career.",
        "An athlete’s career can be affected by accidental doping violations. Stick to trusted supplements, and avoid any that aren’t thoroughly researched or that have unknown ingredients.",
        "Always stay informed about banned substances. Consult with a medical professional or a doping advisor to ensure all supplements you use are safe and legal."
    ],
    "fitness": [
        "For effective fitness training, balance cardio with strength exercises. Include endurance training, but also dedicate time to strength work to build core stability and power.",
        "To build endurance, try interval training or long-distance running. Combine this with weight training to enhance your overall athletic ability.",
        "Aim to work out at least three times a week, and don't forget to stretch before and after your sessions. Stretching helps with flexibility and can prevent injuries.",
        "Remember to allow for rest days between intense workouts. Rest days are crucial for muscle recovery and long-term progress. Pacing yourself will prevent burnout and help you stay consistent."
    ],
    "health": [
        "A balanced diet is crucial for peak performance. Focus on high-quality proteins, complex carbohydrates, and healthy fats to fuel your body effectively.",
        "Stay hydrated, especially during intense training periods. Dehydration can impair performance, so keep a water bottle handy throughout the day.",
        "Good mental health is essential. Consider incorporating practices like mindfulness, meditation, or journaling to manage stress and improve focus.",
        "Sleep is one of the most important factors in recovery. Aim for 7-9 hours of quality sleep each night to allow your body to repair and prepare for the next training session."
    ],
    "motivation": [
        "Staying motivated as an athlete can be challenging. Set small, achievable goals and celebrate each accomplishment to keep yourself encouraged.",
        "Focus on the reasons why you started. Visualize your long-term goals and remind yourself of the progress you've made to stay motivated.",
        "Surround yourself with supportive people, whether teammates, friends, or family. A positive environment can boost your motivation and make training enjoyable.",
        "Consistency is key to success. Even on days when motivation is low, sticking to your routine will help you stay on track and achieve your goals over time."
    ],
    "general": [
        "I'm here to assist you with fitness, health, anti-doping, and motivation. You can ask specific questions in any of these areas!",
        "Ask me about fitness training, dietary advice, motivation tips, or guidance on anti-doping best practices for athletes.",
        "I can help with tips on fitness, health, anti-doping, and motivation. Let me know which area you're interested in!"
    ]
}

# Improved generate_response function with refined templates and contextual hints
def generate_response(user_input):
    if "doping" in user_input or "substance" in user_input:
        response = random.choice(RESPONSE_TEMPLATES["anti_doping"])
    elif "fitness" in user_input or "training" in user_input:
        response = random.choice(RESPONSE_TEMPLATES["fitness"])
    elif "health" in user_input or "nutrition" in user_input or "diet" in user_input:
        response = RESPONSE_TEMPLATES["health"]
    elif "motivate" in user_input or "goal" in user_input or "discouraged" in user_input:
        response = random.choice(RESPONSE_TEMPLATES["motivation"])
    else:
        # Default response if no specific keyword is found
        response = random.choice(RESPONSE_TEMPLATES["general"])

    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_text = request.form["msg"]
    response = generate_response(user_text)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
