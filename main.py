import ollama
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def llm_response():
    if request.method == "GET":
        return render_template('index.html')
    prompt = request.form['prompt']
    response = ollama.generate(
        model='gemma3:270m', 
        prompt=prompt
    )
    return f"<p class='float-right bg-gray-100 px-2 py-1 rounded-lg'>{prompt}</p><br/><p class='mt-4'>{response['response']}</p>"

if __name__ == '__main__':
    app.run(debug=True,port=4006)