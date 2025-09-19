import os
import json
import mlflow
from groq import Groq
from dotenv import load_dotenv
load_dotenv()



api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=api_key)

# Ensure no active MLflow run is left open from previous executions
if mlflow.active_run():
    mlflow.end_run()

expt_name = "GenAI_Prompt_Optimization_Summarization"
mlflow.set_experiment(expt_name)
print(f"\nMLflow Experiment set to: {expt_name}")


# Article to Summarize
sample_article = """
Russia: A Tapestry of History, Power, and Transformation

Russia, the largest country on Earth, boasts a rich and complex history that stretches across centuries and continents. From the emergence of the first Russian states in the medieval period to the establishment of the powerful Tsarist empire, Russia's past is defined by vast lands, shifting borders, and towering figures.

The rise of the Kievan Rus' in the 9th century laid the foundation for the Russian identity, with Christianity playing a pivotal role in shaping its cultural and political trajectory. Centuries later, Peter the Great would push Russia into the realm of European power, modernizing the country through sweeping reforms, the founding of St. Petersburg, and an expansion of Russia's military might. This set the stage for the Romanov dynasty, which ruled for over 300 years, overseeing Russia's growth into an empire spanning from Eastern Europe to the Pacific.

The 20th century marked a dramatic shift with the Russian Revolution of 1917, leading to the collapse of the Tsarist regime and the rise of the Soviet Union. Under Vladimir Lenin and later Joseph Stalin, Russia (now the USSR) transformed into a communist superpower, competing fiercely with the United States during the Cold War. The Soviet period was marked by both significant achievements in science, space exploration (such as launching Sputnik and sending the first man into space), and devastating periods of repression and hardship for many of its people.

With the dissolution of the Soviet Union in 1991, Russia entered a new era, marked by economic turmoil, political reform, and the rise of Vladimir Putin as a central figure in modern Russian politics. Today, Russia is a major global player, rich in natural resources like oil and gas, and deeply engaged in both regional and international affairs. Yet, the country remains a land of contradictions: its long-standing cultural traditions, such as the arts, literature, and ballet, coexist with a modern state navigating geopolitical tensions, challenges to democracy, and the ongoing quest for national identity in a rapidly changing world.

The story of Russia is a story of resilience, ambition, and an ever-evolving vision of itself, from the heart of Moscow to the vast expanses of Siberia, from imperial grandeur to modern-day power struggles. Its influence, both historical and contemporary, continues to shape the course of world events.
"""





# Mock LLM Judge Function (Simulates a real LLM for evaluation)
# In a real scenario, this would be another LLM API call
# that evaluates the summary based on criteria like conciseness, relevance, coherence.
# For simplicity, this mock judge assigns scores based on simple text properties.
def evaluate_summary_with_mock_llm_judge(original_text: str, prompt_used: str, generated_summary: str) -> dict:
    """
    A mock LLM judge to evaluate the quality of a generated summary.
    In a real application, this would involve calling another LLM (e.g., GPT-4, Claude)
    with a specific prompt to rate the summary.
    """
    evaluation_results = {
        "conciseness_score": 0.0,
        "relevance_score": 0.0,
        "coherence_score": 0.0,
        "overall_score": 0.0
    }

    # Simulate conciseness: shorter summaries (within reason) get higher scores
    original_word_count = len(original_text.split())
    summary_word_count = len(generated_summary.split())

    if summary_word_count < 0.2 * original_word_count:
        evaluation_results["conciseness_score"] = 9.0 # Very concise
    elif summary_word_count < 0.4 * original_word_count:
        evaluation_results["conciseness_score"] = 7.0 # Moderately concise
    else:
        evaluation_results["conciseness_score"] = 4.0 # Less concise

    # Simulate relevance: check for keywords from the original text
    keywords = ["AI", "healthcare", "finance", "challenges", "innovation"]
    relevant_keywords_found = sum(1 for kw in keywords if kw.lower() in generated_summary.lower())
    evaluation_results["relevance_score"] = (relevant_keywords_found / len(keywords)) * 10.0

    # Simulate coherence: check for basic sentence structure / length (very basic mock)
    # A real LLM judge would assess flow, grammar, etc.
    if len(generated_summary.split('.')) > 1 and len(generated_summary.split('.')) < 5:
        evaluation_results["coherence_score"] = 8.0
    else:
        evaluation_results["coherence_score"] = 5.0

    evaluation_results["overall_score"] = (evaluation_results["conciseness_score"] +
                                           evaluation_results["relevance_score"] +
                                           evaluation_results["coherence_score"]) / 3.0

    return evaluation_results









# Define Different Prompts to Test
prompts_to_test = [
    {
        "name": "Standard Summary",
        "text": "Summarize the following article concisely: \n{article_text}"
    },
    {
        "name": "Bullet Point Summary",
        "text": "Provide a summary of the following article in 3-5 bullet points, focusing on key challenges and benefits: \n{article_text}"
    },
    {
        "name": "One Sentence Summary",
        "text": "Condense the following article into a single, comprehensive sentence: \n{article_text}"
    },
    {
        "name": "Detailed Summary",
        "text": "Write a detailed summary of the following article, ensuring all major aspects are covered: \n{article_text}"
    }
]



# Run Experiments
for i, prompt_data in enumerate(prompts_to_test):
    prompt_name = prompt_data["name"]
    raw_prompt_template = prompt_data["text"]
    
    # Fill in the article text into the prompt template
    current_prompt = raw_prompt_template.format(article_text=sample_article)
    print(f"\n--- Running Experiment for Prompt: '{prompt_name}' ({i+1}/{len(prompts_to_test)}) ---")

    with mlflow.start_run(run_name=prompt_name):
        mlflow.log_param("prompt_strategy", prompt_name)
        mlflow.log_param("raw_prompt_template", raw_prompt_template)
        mlflow.log_param("full_input_prompt", current_prompt)
        mlflow.log_param("model_name", "llama3-8b-8192")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_tokens", 500)


        # Original Article - Log Artifact
        with open("original_article.txt", "w") as f:
            f.write(sample_article)
        mlflow.log_artifact("original_article.txt")

        try:
            # --- Call Groq API ---
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": current_prompt}],
                model = "llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=500
            )
            generated_text = chat_completion.choices[0].message.content
            print(f"Generated Summary ({prompt_name}):\n{generated_text}\n")

            # --- Log Generated Text as Artifact ---
            with open(f"generated_summary_{prompt_name.replace(' ', '_')}.txt", "w") as f:
                f.write(generated_text)
            mlflow.log_artifact(f"generated_summary_{prompt_name.replace(' ', '_')}.txt")

            # --- Log Basic Metric ---
            mlflow.log_metric("generated_summary_length", len(generated_text.split()))

            # --- Evaluate with LLM Judge and Log Metrics ---
            evaluation_scores = evaluate_summary_with_mock_llm_judge(sample_article, current_prompt, generated_text)
            for metric_name, score in evaluation_scores.items():
                mlflow.log_metric(f"judge_{metric_name}", score)
            print(f"Judge Scores: {evaluation_scores}")

            # --- Log Full API Response as Artifact (Optional but useful) ---
            with open(f"chat_completion_{prompt_name.replace(' ', '_')}.json", "w") as f:
                json.dump(chat_completion.to_dict(), f, indent=4)
            mlflow.log_artifact(f"chat_completion_{prompt_name.replace(' ', '_')}.json")

        except Exception as e:
            print(f"❌ Error during generation for '{prompt_name}': {e}")
            mlflow.log_param("generation_error", str(e))
            mlflow.set_tag("status", "failed")
            continue # Continue to next prompt if one fails

print("\n✅ All MLflow runs completed for prompt optimization.")