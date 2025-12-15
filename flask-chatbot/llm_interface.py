import os
import json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from config import read_configs
import requests
from datetime import datetime

# Define tools for the agent
@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def generate_image(prompt: str, model_name: str = "dall-e-3") -> str:
    """Generate an image based on a text prompt using OpenAI's DALL-E.

    Args:
        prompt: The description of the image to generate
        model_name: The model to use (dall-e-3 or dall-e-2)

    Returns:
        URL of the generated image or error message
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OpenAI API key not found"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model_name,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            image_url = response.json()["data"][0]["url"]
            return f"Image generated successfully! URL: {image_url}\n\n![Generated Image]({image_url})"
        else:
            return f"Error generating image: {response.json().get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

class ChatBackend():

    def load_conversation_state(self, state_file):
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
        return []

    def __init__(self, config_address="config.json"):
        configs = read_configs(config_address)
        os.environ["OPENAI_API_KEY"] = configs["chatbot"]["OPENAI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = configs["chatbot"]["GOOGLE_API_KEY"]
        # self.model = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)
        self.state_path = configs["chatbot"].get("CONV_LOG_PATH", "conversation_state_dev.json")
        self.history = self.load_conversation_state(self.state_path)
        # system_prompt = SystemMessage(content="You are Coder Agent, an expert AI assistant who writes clean and easy to understand code. At each step you need write the best code possible. different options are not necessary. Don't add comments to code. The code should be production ready. Demos, incomplete code or code that requires further work is not acceptable. When a piece of code is provided, you should not change it unless the user asks you to do so. If the user asks you to change a piece of code, you should only change necessary part of the code and not the rest of the code.")
        system_prompt = SystemMessage(content="Your are a helpful AI assistant. Be short and concise in your answers.")
        self.history_langchain_format = [system_prompt]
        for msg in self.history:
            if msg["role"] == "user":
                self.history_langchain_format.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                self.history_langchain_format.append(AIMessage(content=msg["content"]))
        with open('/home/raha/genai-course-codebase/flask-chatbot/static/settings.json', 'r') as f:
            data = json.load(f)["language_models"]
        self.model_dict = {}
        for d in data:
            self.model_dict[d["name"]] = d["provider"]

        # Initialize tools and memory for agent
        self.tools = [get_current_time, calculate, generate_image]
        self.memory = MemorySaver()
        self.agent = None
        self.agent_config = {"configurable": {"thread_id": "default"}}


    def save_conversation_state(self, history):
        with open(self.state_path, "w") as f:
            json.dump(history, f, indent=2)

    def predict(self, message, model_name=None):
        if model_name is None:
            model_name = getattr(self, "default_model", "gpt-4.1")
        print("Using model:", model_name)
        provider = self.model_dict[model_name]

        # Create the appropriate model
        if provider == "google":
            model = ChatGoogleGenerativeAI(model=model_name, streaming=True)
        elif provider == "ollama":
            model = OllamaLLM(model=model_name, streaming=True)
        else:
            model = ChatOpenAI(model=model_name, streaming=True)

        # Determine which tools to use based on provider
        # OpenAI models support all tools including image generation
        # Other providers get limited tools
        if provider == "openai":
            tools_to_use = self.tools
        else:
            # For non-OpenAI models, exclude image generation
            tools_to_use = [get_current_time, calculate]

        # Create agent with the model and tools
        self.agent = create_react_agent(model, tools_to_use, checkpointer=self.memory)

        # Add user message to history
        self.history_langchain_format.append(HumanMessage(content=message))

        # Stream agent response
        response = ""
        try:
            for chunk in self.agent.stream(
                {"messages": self.history_langchain_format},
                self.agent_config,
                stream_mode="values"
            ):
                if "messages" in chunk and len(chunk["messages"]) > 0:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "content") and last_message.content:
                        # Check if this is a new message or continuation
                        current_content = last_message.content
                        if len(current_content) > len(response):
                            delta = current_content[len(response):]
                            response = current_content
                            if delta:
                                yield delta
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
            response = error_msg

        # Update history
        self.history_langchain_format.append(AIMessage(content=response))
        self.history = self.history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        self.save_conversation_state(self.history)

    def reset_conversation(self):
        self.save_conversation_state([])

    def get_history(self):
        return self.history