import ollama
from quart import Quart, render_template, request, Response
from datetime import datetime
import json

app = Quart(__name__)
client = ollama.AsyncClient()

# ==========================================
# 1. DEFINE YOUR TOOLS
# ==========================================
def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    return a + b

def get_current_time(time_format: str = "%H:%M:%S") -> str:
    """
    Returns the current local time. 

    Args:
        time_format (str): Standard strftime format string (e.g., "%H:%M:%S" or "%I:%M %p"). Defaults to "%H:%M:%S".
    """
    return datetime.now().strftime(time_format)

# ==========================================
# 2. CENTRAL TOOL REGISTRY
# ==========================================
# Just add any new python function to this dictionary to instantly register it!
AVAILABLE_TOOLS = {
    'add_two_numbers': add_two_numbers,
    'get_current_time': get_current_time
}

@app.route("/", methods=["GET"])
async def home():
    try:
        response = await client.list()
        models = response.get('models', [])
    except Exception as e:
        models = []
    return await render_template("index.html", models=models)

@app.route("/chat", methods=["POST"])
async def llm_response():
    data = await request.get_json() or await request.form
    prompt = data.get('prompt', '')
    model = data.get('model') or 'llama3.2'
    messages = data.get('messages', [])
    
    if not messages or messages[-1].get('content') != prompt:
        messages.append({'role': 'user', 'content': prompt})

    can_use_tools = any(name in model.lower() for name in ['llama3.1', 'llama3.2', 'qwen3.5:4b', 'mistral', 'command-r', 'qwen2.5'])
    
    async def generate_stream():
        full_assistant_content = ""
        try:
            kwargs = {'model': model, 'messages': messages, 'stream': True}
            if can_use_tools:
                # Dynamically pass all Python function objects registered in our map
                kwargs['tools'] = list(AVAILABLE_TOOLS.values())

            stream = await client.chat(**kwargs)
            
            tool_calls = []
            async for chunk in stream:
                if chunk.message.content:
                    full_assistant_content += chunk.message.content
                    yield chunk.message.content
                
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            if tool_calls:
                # Establish assistant intent context
                assistant_msg = {'role': 'assistant', 'tool_calls': tool_calls}
                if full_assistant_content:
                    assistant_msg['content'] = full_assistant_content
                messages.append(assistant_msg)
                
                # Execute every requested tool cleanly using our registry lookup
                for tool in tool_calls:
                    tool_name = tool.function.name
                    args = tool.function.arguments

                    if tool_name in AVAILABLE_TOOLS:
                        # Call the registered python function dynamically matching arguments
                        result = AVAILABLE_TOOLS[tool_name](**args)
                        
                        yield f"\n\n[Tool Called: {tool_name}({args}) -> Result: {result}]\n\n"
                        
                        # Add a dedicated message for every distinct tool block executed
                        messages.append({
                            'role': 'tool',
                            'name': tool_name,  # Essential for LLM debugging alignment
                            'content': str(result)
                        })
                    else:
                        yield f"\n\n[Error: Requested tool '{tool_name}' is unregistered]\n\n"

                # Send back to Ollama to synthesize the tool answers into structural text
                final_stream = await client.chat(model=model, messages=messages, stream=True)
                final_assistant_content = ""
                async for chunk in final_stream:
                    if chunk.message.content:
                        final_assistant_content += chunk.message.content
                        yield chunk.message.content
                
                messages.append({'role': 'assistant', 'content': final_assistant_content})
            else:
                messages.append({'role': 'assistant', 'content': full_assistant_content})

        except Exception as e:
            print(f"Stream Error: {str(e)}")
            yield f"\n[Server Error: {str(e)}]"

    return Response(generate_stream(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=4006)
