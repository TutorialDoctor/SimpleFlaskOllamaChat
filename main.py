import ollama
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def  home():
    response = ollama.list()

    return render_template("index.html",models=response['models'])

@app.route("/chat", methods=["POST"])
def llm_response():
    prompt = request.form['prompt']
    model = request.form['model']
    response = ollama.generate(
        model= model or 'gemma3:270m', 
        prompt=prompt
    )
    return f"""<div>
                            <span
                                class="text-right bg-zinc-800 text-white p-2 rounded-lg flex w-max max-w-[75%] flex-col rounded-lg px-3 py-2 ml-auto">{prompt}</span>
                            <span
                                class="text-left my-3 bg-gray-200 text-gray-700 p-2 rounded-lg flex w-max max-w-[75%] flex-col rounded-lg px-3 py-2">{response['response']}</span>
                        </div>"""

if __name__ == '__main__':
    app.run(debug=True,port=4006)