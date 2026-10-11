import ollama
from quart import Quart, render_template, request

app = Quart(__name__)
client = ollama.AsyncClient()

@app.route("/", methods=["GET"])
async def home():
    response = await client.list()
    return await render_template("index.html", models=response['models'])

@app.route("/chat", methods=["POST"])
async def llm_response():
    form = await request.form
    prompt = form.get('prompt', '')
    model = form.get('model') or 'gemma3:270m'

    response = await client.generate(
        model=model, 
        prompt=prompt
    )

    # Return ONLY the AI response. The prompt is already rendered on the frontend.
    return f"""<div>
        <span class="text-left my-3 bg-gray-200 text-gray-700 p-2 rounded-lg flex w-max max-w-[75%] flex-col rounded-lg px-3 py-2">{response['response']}</span>
    </div>"""

if __name__ == '__main__':
    app.run(debug=True, port=4006)