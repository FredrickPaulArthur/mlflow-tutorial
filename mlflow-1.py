"""
What are Experiments?

An experiment typically refers to a controlled process where a specific hypothesis or objective is tested by manipulating certain
variables (data, model parameters, algorithms, etc.) and observing the outcomes. These experiments help in understanding, validating
and improving models, as well as in drawing conclusions based on the data.
"""

import os
import json
import mlflow
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

if mlflow.active_run():
    mlflow.end_run()


mlflow.set_experiment("Groq_BananaCake_Experiment-llama_versatile")


# Start run
with mlflow.start_run():
    model = "llama-3.3-70b-versatile"
    temp = 0.7
    max_toks = 5000
    prompt = "Explain me how to make Banana Cake from scratch?"
    timeout = 10

    # Log Parameters
    mlflow.log_param("model_name", model)
    mlflow.log_param("temperature", temp)
    mlflow.log_param("max_tokens", max_toks)
    mlflow.log_param("prompt", prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                { "role": "user", "content": prompt }
            ],
            model = model,
            temperature=temp,
            max_tokens=max_toks
        )
        generated_text = chat_completion.choices[0].message.content
        print(f"Generated Text: {generated_text}")

        # Log Metric
        mlflow.log_metric("generated_text_length", len(generated_text))

        # Save and Log text
        with open("generated_text.txt", "w") as f:
            f.write(generated_text)
        mlflow.log_artifact("generated_text.txt")

        # Save and Log full object
        with open("chat_completion.json", "w") as f:
            json.dump(chat_completion.to_dict(), f, indent=4)
        mlflow.log_artifact("chat_completion.json")

    except Exception as e:
        print(f"❌ Error during generation: {e}")

print("✅ MLFlow run completed.")