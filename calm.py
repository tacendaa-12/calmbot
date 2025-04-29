import os
import re
import random
import json
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import warnings
warnings.filterwarnings('ignore')

class MentalHealthChatbot:
    def __init__(self):
        self.load_knowledge_base()
        <script src="https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js"></script>

        
        
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform([q for intent in self.knowledge_base["intents"] for q in intent["patterns"]])
        

        self.user_state = {
            "name": None,
            "mood": None,
            "crisis_detected": False,
            "session_start": datetime.now(),
            "interaction_count": 0,
            "topics_discussed": set()
        }
        

        self.crisis_keywords = [
            "suicide", "kill myself", "end my life", "don't want to live", 
            "hurt myself", "self-harm", "die", "death", "dangerous", "emergency"
        ]
        

        self.conversation_history = []
        

        self.grounding_techniques = [
            "Try the 5-4-3-2-1 technique: Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
            "Take a deep breath in for 4 counts, hold for 7 counts, and exhale for 8 counts. Repeat this a few times.",
            "Place your feet firmly on the ground and focus on the sensation of the floor beneath you.",
            "Hold an ice cube in your hand and focus on the cold sensation as it melts.",
            "Slowly count backward from 100 by 7s (100, 93, 86...)."
        ]
        
        print("Mental Health Support Chatbot initialized. Type 'exit' to end the conversation.")
        print("IMPORTANT: This chatbot is for educational purposes only and is not a substitute for professional mental health care.")
        print("If you're experiencing a mental health emergency, please call your local emergency number or a crisis hotline.")
        print("US National Suicide Prevention Lifeline: 988 or 1-800-273-8255\n")
        
    def load_knowledge_base(self):
        
        self.knowledge_base = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "what's up"],
                    "responses": [
                        "Hello! I'm here to support you with your mental health. How are you feeling today?",
                        "Hi there! I'm a mental health support chatbot. How can I help you today?",
                        "Hello! I'm here to listen and support you. Would you like to tell me how you're feeling?"
                    ]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["bye", "see you", "goodbye", "later", "exit", "quit"],
                    "responses": [
                        "Take care of yourself. Remember that seeking help is a sign of strength.",
                        "Goodbye! Don't forget to practice self-care regularly.",
                        "Take care! Remember, it's okay to reach out for support when you need it."
                    ]
                },
                {
                    "tag": "gratitude",
                    "patterns": ["thank you", "thanks", "helpful", "appreciate", "grateful"],
                    "responses": [
                        "You're welcome! I'm here to support you.",
                        "I'm glad I could help in some way.",
                        "It's my purpose to be here for you. I'm happy if I've been helpful."
                    ]
                },
                {
                    "tag": "anxiety",
                    "patterns": [
                        "feeling anxious", "anxiety", "worried", "panic", "stress", "can't calm down",
                        "heart racing", "nervous", "overwhelmed", "fear", "scared"
                    ],
                    "responses": [
                        "I'm sorry you're feeling anxious. Let's try a grounding technique: {grounding}",
                        "Anxiety can be really challenging. Deep breathing can help - try breathing in slowly for 4 counts, hold for 7, and exhale for 8.",
                        "When you're feeling anxious, it can help to focus on the present moment. Could you tell me 5 things you can see right now?"
                    ]
                },
                {
                    "tag": "depression",
                    "patterns": [
                        "depressed", "sad", "hopeless", "no energy", "tired all the time",
                        "don't enjoy anything", "worthless", "empty", "numb", "lonely"
                    ],
                    "responses": [
                        "I'm sorry you're feeling this way. Depression can make everything feel more difficult. Have you spoken to a mental health professional about these feelings?",
                        "It sounds like you're going through a really tough time. Remember that depression often lies to us about our worth and possibilities. What's one small thing you could do today that might bring even a tiny bit of relief?",
                        "Those feelings of depression can be so heavy. Sometimes taking one small step, like getting some fresh air or reaching out to a friend, can help a little bit. Would either of those be possible for you today?"
                    ]
                },
                {
                    "tag": "sleep",
                    "patterns": [
                        "can't sleep", "insomnia", "trouble sleeping", "nightmares",
                        "sleep problems", "tired", "exhausted", "fatigue"
                    ],
                    "responses": [
                        "Sleep difficulties can really affect our mental health. Have you tried establishing a consistent sleep routine? Going to bed and waking up at the same time each day can help.",
                        "I'm sorry you're having trouble sleeping. Some things that might help include avoiding screens an hour before bed, keeping your bedroom cool and dark, and practicing relaxation techniques.",
                        "Sleep problems are common with many mental health conditions. Have you mentioned these sleep issues to your doctor? They might have some specific recommendations for your situation."
                    ]
                },
                {
                    "tag": "mindfulness",
                    "patterns": [
                        "mindfulness", "meditation", "relax", "calm", "present moment",
                        "breathing exercises", "grounding", "peace", "zen"
                    ],
                    "responses": [
                        "Mindfulness can be a powerful tool for mental health. One simple practice is to focus on your breath for a few minutes, noticing when your mind wanders and gently bringing it back.",
                        "Practicing mindfulness helps us stay in the present moment rather than worrying about the future or dwelling on the past. Would you like a simple mindfulness exercise to try?",
                        "Even a few minutes of mindfulness each day can make a difference. Try focusing on one of your senses - what can you hear right now, from the loudest to the quietest sounds?"
                    ]
                },
                {
                    "tag": "self_care",
                    "patterns": [
                        "self care", "self-care", "take care of myself", "burned out",
                        "exhausted", "overwhelmed", "need a break", "stress"
                    ],
                    "responses": [
                        "Self-care is essential for mental health. This includes basics like sleep, nutrition, and exercise, but also activities that bring you joy and relaxation. What self-care activities do you enjoy?",
                        "When we're struggling, self-care can be one of the first things we neglect. Could you identify one small act of self-care you could do today?",
                        "Self-care isn't selfish - it's necessary. Think of it as refilling your cup so you have something to give. What's one small way you could show yourself some kindness today?"
                    ]
                },
                {
                    "tag": "crisis",
                    "patterns": [
                        "want to die", "kill myself", "suicide", "end it all", "self harm", 
                        "cut myself", "hurt myself", "no point in living", "better off without me",
                        "want to end my life"
                    ],
                    "responses": [
                        "I'm really concerned about what you're sharing. If you're in immediate danger, please call emergency services (911 in the US) or a crisis line like the 988 Suicide & Crisis Lifeline (call or text 988). Would you like me to provide more crisis resources?",
                        "What you're experiencing sounds really serious, and I want to make sure you get the support you need. The 988 Suicide & Crisis Lifeline is available 24/7 by calling or texting 988. They have trained counselors who can help. Would it be possible for you to reach out to them?",
                        "I'm concerned about your safety right now. Please consider calling the 988 Suicide & Crisis Lifeline (call or text 988) or going to your nearest emergency room. Is there someone nearby who could stay with you right now?"
                    ]
                },
                {
                    "tag": "therapy",
                    "patterns": [
                        "therapy", "therapist", "counseling", "counselor", "psychologist",
                        "psychiatrist", "mental health professional", "treatment"
                    ],
                    "responses": [
                        "Therapy can be a really helpful tool for many people. There are different types of therapy like CBT, DBT, and psychodynamic therapy. Have you worked with a therapist before?",
                        "Finding the right therapist can take time, but it's worth the effort. It's important to find someone you feel comfortable with. Would you like some tips on how to find a therapist?",
                        "Many people find therapy beneficial for working through challenges and developing coping skills. If you're considering therapy, your primary care doctor might be able to provide a referral."
                    ]
                },
                {
                    "tag": "resources",
                    "patterns": [
                        "resources", "help", "support", "hotline", "crisis line", 
                        "where can I get help", "support groups"
                    ],
                    "responses": [
                        "There are many mental health resources available. In the US, the 988 Suicide & Crisis Lifeline is available 24/7 by calling or texting 988. The Crisis Text Line is available by texting HOME to 741741. Many communities also have local support groups and services.",
                        "For immediate support, crisis lines like 988 in the US can be helpful. For ongoing support, you might consider therapy, support groups, or online communities. Would you like information about a specific type of resource?",
                        "Mental health resources include crisis lines, therapy, psychiatry, support groups, and self-help resources. Is there a particular type of support you're looking for?"
                    ]
                },
                {
                    "tag": "coping_skills",
                    "patterns": [
                        "coping", "coping skills", "deal with", "handle", "manage", 
                        "how to cope", "techniques", "strategies"
                    ],
                    "responses": [
                        "Developing healthy coping skills is important for mental health. Some effective coping strategies include deep breathing, physical exercise, journaling, talking to a trusted friend, and practicing mindfulness. What coping skills have worked for you in the past?",
                        "Coping skills can be different for everyone. Some people find physical activities helpful, while others prefer creative outlets or social connection. It can be helpful to have a variety of coping skills for different situations. Would you like to brainstorm some coping strategies?",
                        "When developing coping skills, it's important to find strategies that work for you. This might include grounding techniques, distraction activities, ways to express emotions, or methods to challenge negative thoughts. What kinds of activities help you feel even a little bit better when you're struggling?"
                    ]
                },
                {
                    "tag": "unknown",
                    "patterns": [],
                    "responses": [
                        "I'm not sure I fully understand. Could you tell me more about what you're experiencing?",
                        "I want to make sure I'm being helpful. Could you share a bit more about your situation?",
                        "I'm here to support you, but I'm having trouble understanding. Could you rephrase that or provide more details?"
                    ]
                }
            ]
        }
    
    def preprocess_text(self, text):
        """Clean and normalize text input"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def detect_intent(self, user_input):
        """Match user input to an intent in our knowledge base"""
        preprocessed_input = self.preprocess_text(user_input)
        

        for keyword in self.crisis_keywords:
            if keyword in preprocessed_input:
                self.user_state["crisis_detected"] = True
                return "crisis"

        user_vector = self.vectorizer.transform([preprocessed_input])

        similarities = cosine_similarity(user_vector, self.X)
  
        if np.max(similarities) < 0.3:
            return "unknown"
        
        idx = np.argmax(similarities)
        count = 0
        for intent in self.knowledge_base["intents"]:
            for pattern in intent["patterns"]:
                if count == idx:
                    return intent["tag"]
                count += 1
                
        return "unknown"
    
    def get_response(self, intent_tag):
        """Get a response based on the detected intent"""
        for intent in self.knowledge_base["intents"]:
            if intent["tag"] == intent_tag:
                response = random.choice(intent["responses"])
                
                if "{grounding}" in response:
                    technique = random.choice(self.grounding_techniques)
                    response = response.replace("{grounding}", technique)
                    
                self.user_state["topics_discussed"].add(intent_tag)
                return response
                
        return "I'm not sure how to respond to that. Could you rephrase?"
    
    def monitor_user_state(self, user_input):
        """Track user state and detect potential issues"""

        self.user_state["interaction_count"] += 1
        
        if self.user_state["name"] is None:
            name_match = re.search(r'my name is (\w+)', user_input.lower())
            if name_match:
                self.user_state["name"] = name_match.group(1).capitalize()
        
        mood_words = {
            "positive": ["happy", "good", "great", "better", "well", "okay", "fine", "alright"],
            "negative": ["sad", "bad", "terrible", "worse", "depressed", "anxious", "worried", "stressed"],
            "neutral": ["neutral", "ok", "so-so", "meh", "average"]
        }
        
        for mood, words in mood_words.items():
            for word in words:
                if word in user_input.lower().split():
                    self.user_state["mood"] = mood
                    break
    
    def generate_check_in(self):
        """Generate check-in messages based on user state"""

        if self.user_state["crisis_detected"] and self.user_state["interaction_count"] > 3:
            return "I notice you've mentioned some concerning thoughts. I want to remind you that help is available. The 988 Suicide & Crisis Lifeline (988) is available 24/7. Would it be possible for you to reach out to them?"

        if "depression" in self.user_state["topics_discussed"] and self.user_state["interaction_count"] % 5 == 0:
            return "Depression can feel overwhelming. Have you been able to talk to a healthcare provider about what you're experiencing?"
            
        if "anxiety" in self.user_state["topics_discussed"] and self.user_state["interaction_count"] % 5 == 0:
            return "We've talked about anxiety today. Would it be helpful to try another grounding technique?"
        
        if self.user_state["interaction_count"] == 10:
            return "We've been talking for a while now. How are you feeling about our conversation? Is it helpful for you?"
        
        return None
    
    def generate_personalized_response(self, base_response, user_input):
        """Add personalization to the response"""
        response = base_response
 
        if self.user_state["name"] and random.random() < 0.3:
            name_phrases = [
                f"{self.user_state['name']}, ",
                f" {self.user_state['name']}.",
                f"I hear you, {self.user_state['name']}. "
            ]
            response = random.choice(name_phrases) + response
        
  
        check_in = self.generate_check_in()
        if check_in:
            response += f" {check_in}"
        
        return response
    
    def chat(self):
        """Main chat loop"""
        user_input = input("You: ")
        
        while user_input.lower() not in ['exit', 'quit', 'bye']:

            self.conversation_history.append({"user": user_input})
            
            self.monitor_user_state(user_input)

            intent = self.detect_intent(user_input)
            
 
            base_response = self.get_response(intent)
            

            final_response = self.generate_personalized_response(base_response, user_input)
            

            self.conversation_history.append({"bot": final_response})
            

            print(f"Bot: {final_response}")
            

            if self.user_state["crisis_detected"]:
                print("\nIMPORTANT: If you're experiencing a crisis, please reach out for help:")
                print("- Call or text 988 (US Suicide & Crisis Lifeline)")
                print("- Text HOME to 741741 (Crisis Text Line)")
                print("- Call 911 or go to your nearest emergency room")
                
    
            user_input = input("You: ")
        
        print(f"Bot: {self.get_response('goodbye')}")

class AIEnhancedMentalHealthChatbot(MentalHealthChatbot):
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_ai = self.api_key is not None
        
        if not self.use_ai:
            print("No API key found. Running in rule-based mode only.")
            print("For enhanced AI capabilities, set the OPENAI_API_KEY environment variable.")
        else:
            print("AI enhancement active: Using AI to generate more personalized responses.")
    
    def get_ai_response(self, user_input, conversation_history=None):
        """Use an AI model to generate a response"""
        if not self.use_ai:
            return None
            
        try:
    
            messages = [{"role": "system", "content": 
                        """You are a mental health support chatbot. Provide compassionate, helpful responses.
                        Focus on supportive listening, validating emotions, and suggesting coping strategies.
                        Never diagnose or provide medical advice.
                        For any mention of self-harm or suicide, emphasize the importance of seeking immediate professional help.
                        Keep responses conversational, warm, and relatively brief (2-4 sentences).
                        Do not pretend to be a human or a therapist - be clear about being a support chatbot."""}]
            
            if conversation_history:
                for i in range(min(5, len(conversation_history))):
                    if "user" in conversation_history[-i-1]:
                        messages.append({"role": "user", "content": conversation_history[-i-1]["user"]})
                    elif "bot" in conversation_history[-i-1]:
                        messages.append({"role": "assistant", "content": conversation_history[-i-1]["bot"]})
            
            messages.append({"role": "user", "content": user_input})
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                return None
                
        except Exception as e:
            print(f"Error calling AI API: {e}")
            return None
    
    def chat(self):
        """Enhanced chat loop with AI capabilities"""
        print("Type 'exit' to end the conversation.")
        user_input = input("You: ")
        
        while user_input.lower() not in ['exit', 'quit', 'bye']:

            self.conversation_history.append({"user": user_input})
        
            self.monitor_user_state(user_input)
            
            if any(keyword in user_input.lower() for keyword in self.crisis_keywords):
                self.user_state["crisis_detected"] = True
                intent = "crisis"
                response = self.get_response(intent)
                
            else:
                
                ai_response = self.get_ai_response(user_input, self.conversation_history) if self.use_ai else None
                
                if ai_response:
                    response = ai_response
                else:
                
                    intent = self.detect_intent(user_input)
                    response = self.get_response(intent)
   
            final_response = self.generate_personalized_response(response, user_input)
            
            self.conversation_history.append({"bot": final_response})
            
            print(f"Bot: {final_response}")
            
            if self.user_state["crisis_detected"]:
                print("\nIMPORTANT: If you're experiencing a crisis, please reach out for help:")
                print("- Call or text 988 (US Suicide & Crisis Lifeline)")
                print("- Text HOME to 741741 (Crisis Text Line)")
                print("- Call 911 or go to your nearest emergency room")
                
            user_input = input("You: ")
        
        print(f"Bot: {self.get_response('goodbye')}")

def main():
  
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("=== Mental Health Support Chatbot ===")
    print("DISCLAIMER: This is a demonstration and not a substitute for professional mental health care.")
    print("If you're experiencing a mental health emergency, please contact emergency services or a crisis hotline.")
    
    if api_key:
        print("AI enhancement active")
        chatbot = AIEnhancedMentalHealthChatbot(api_key)
    else:
        print("Running in rule-based mode")
        print("For AI enhancement, set the OPENAI_API_KEY environment variable")
        chatbot = MentalHealthChatbot()
    
    chatbot.chat()

if __name__ == "__main__":
    main()