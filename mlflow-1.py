"""
What are Experiments?

An experiment typically refers to a controlled process where a specific hypothesis or objective is tested by manipulating certain
variables (data, model parameters, algorithms, etc.) and observing the outcomes. These experiments help in understanding, validating
and improving models, as well as in drawing conclusions based on the data.

Parameters  -   Parameters are the input variables that influence the model training. These are usually hyperparameters that 
                you pass to the model or algorithm during training, such as learning rate, batch size, number of epochs, etc.
                    mlflow.log_param("learning_rate", 0.01)

Metrics     -   Metrics are numeric measurements that reflect the performance or evaluation of your model during training or
                testing. These metrics can include things like accuracy, loss, F1 score, or any other performance-related
                quantity that changes over time.
                    mlflow.log_metric("accuracy", 0.92)
                    mlflow.log_metric("loss", 0.3)

Artifacts   -   Artifacts are the outputs or files that your model generates during or after training. These can be the model
                itself, logs, trained weights, visualizations, or any other file that can be useful for future analysis or model
                deployment.
                    mlflow.log_artifact("model.pkl")
                    mlflow.log_artifact("confusion_matrix.png")

"""

import os
import json
import mlflow
import dvc
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

if mlflow.active_run():
    mlflow.end_run()



# 1. CREATE EXPERIMENT
mlflow.set_experiment("Groq_BananaCake_Experiment-llama_versatile_")

# Team specific credentials - for Collaboration
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")


# Publishing to Local Repository
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Publishing to Centralized DagsHub Server
# print(mlflow.set_tracking_uri("MLFLOW_TRACKING_URI"))


# 2. START RUN - can also specify run_name, run_id, expt_id, etc.
with mlflow.start_run():
    model = "llama-3.3-70b-versatile"
    temp = 0.7
    max_toks = 5000
    prompt = "Explain me how to make Banana Cake from scratch?"
    timeout = 10

    # Use this to log an Entire Experiment with some tags
    # Can use this also in a separate script mentioning the Experiment ID.
    mlflow.set_experiment_tag("team", "data-science")
    mlflow.set_experiment_tag("goal", "model-optimization")

    # Use this to log a single run in an Experiment
    mlflow.set_tag("project", "text-generation")
    mlflow.set_tag("dataset", "none")
    mlflow.set_tag("experiment_type", "groq_api_calling")
    mlflow.set_tag("release.version", "1.0.0")


    # 3. LOG PARAMETERS
    mlflow.log_param("model_name", model)
    mlflow.log_param("temperature", temp)
    mlflow.log_param("max_tokens", max_toks)
    mlflow.log_param("prompt", prompt)
    mlflow.log_param("llm_provider", "groq.com")
    mlflow.log_param("random_state", 2049)
    # mlflow.log_param("precision", precision)
    # mlflow.log_param("recall", recall)
    # mlflow.log_param("f1_score", cls_report['macro avg'']['f1-score'])
    # mlflow.log_param("recall_class_0", cls_report['0']['recall'])
    # mlflow.log_param("recall_class_1", cls_report['1']['recall'])
    # mlflow.log_param("max_depth", max_depth)

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


    # 4. LOG METRICS
        mlflow.log_metric("generated_text_length", len(generated_text))
        mlflow.log_metric("input_prompt_length", len(prompt))

    # 5. LOG ARTIFACTS
        with open("generated_text.txt", "w") as f:      # Save and Log text
            f.write(generated_text)
        with open("chat_completion.json", "w") as f:    # Save and Log full object
            json.dump(chat_completion.to_dict(), f, indent=4)
        mlflow.log_artifact("generated_text.txt")
        mlflow.log_artifact("chat_completion.json")

        # Requirements
        mlflow.log_artifact("requirements.txt")
        mlflow.log_artifact("original_article.txt")
        
    # 6. DATASET
        # mlflow.log_artifact("path/to/your/dataset.csv", artifact_path="datasets/v1")
        
    # 7. MODEL
        # mlflow.sklearn.log_model(model_1, name="Logistic Regression")
        # mlflow.xgboost.log_model(model_2, name="XGBClassifier")


    # 8. REGISTER THE MODEL - of a specific run
        # model_name = "XGB-Smote"
        # run_id = input("Enter Run ID of model to register: ")
        # model_uri = f"runs:/{run_id}/{model_name}"
        # print(f"Model with name {model_name} has been registered with URI - {model_uri}")

        # result = mlflow.register_model(model_uri, model_name)


    # 9. LOAD THE MODEL and MAKE PREDICTIONS
        # Development model's Alias is maintained as "@challenger" in UI.
        # model_version = 1
        # model_uri = f"models:/{model_name}/{model_version}"
        # model_uri_challenger = f"models:/{model_name}@challenger"       # For loading the model with alias - @challenger

        # loaded_model = mlflow.xgboost.load_model(model_uri_challenger)
        # y_preds = loaded_model.predict(X_test)
        # print("\nYour predictions: ", y_preds[:4])


    # 10. For Production
    #     # Change the model's alias to "@champion" in the UI, then use it.
    #     dev_model_uri = f'models:/{model_name}@champion'          # For loading the model with alias - @champion
    #     prod_model_name = 'anomaly-detection-prod'
    #     client = mlflow.MlflowClient()
    #     client.copy_model_version(
    #         src_model_uri=dev_model_uri, 
    #         dst_name=prod_model_name
    #     )

        # And then you deploy the model using Docker, AWS, Databricks, etc
        # MLFlow APIs are available

    except Exception as e:
        print(f"❌ Error during generation: {e}")

print("✅ MLFlow run completed.")