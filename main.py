import ollama
from quart import Quart, render_template, request, Response

app = Quart(__name__)
client = ollama.AsyncClient()

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
    form = await request.form
    prompt = form.get('prompt', '')
    model = form.get('model') or 'llama3.2'

    messages = [{'role': 'user', 'content': prompt}]
    can_use_tools = any(name in model.lower() for name in ['llama3.1', 'llama3.2', 'qwen3.5:4b', 'mistral', 'command-r'])
    
    async def generate_stream():
        try:
            kwargs = {'model': model, 'messages': messages, 'stream': True}
            if can_use_tools:
                kwargs['tools'] = [add_two_numbers]

            stream = await client.chat(**kwargs)
            
            tool_calls = []
            async for chunk in stream:
                # Stream text content live as it arrives
                if chunk.message.content:
                    yield chunk.message.content
                
                # Capture any tool calls if triggered
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            # 2. If a tool was triggered during the stream, process it and stream the final answer
            if tool_calls:
                messages.append({'role': 'assistant', 'tool_calls': tool_calls})
                
                tool_output_str = ""
                for tool in tool_calls:
                    if tool.function.name == 'add_two_numbers':
                        args = tool.function.arguments
                        result = add_two_numbers(**args)
                        tool_output_str = f"Result of {tool.function.name}({args}) is {result}"
                        yield f"\n\n[Tool Called: {tool.function.name}({args}) -> Result: {result}]\n\n"
                
                messages.append({
                    'role': 'tool',
                    'content': tool_output_str
                })
                
                # 3. Stream the final summary response based on the tool result
                final_stream = await client.chat(model=model, messages=messages, stream=True)
                async for chunk in final_stream:
                    if chunk.message.content:
                        yield chunk.message.content

        except Exception as e:
            print(f"Stream Error: {str(e)}")
            yield f"\n[Server Error: {str(e)}]"

    return Response(generate_stream(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=4006)