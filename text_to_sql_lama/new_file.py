#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()  # Ensure your .env file contains your LAMINI_API_KEY

import lamini
import logging
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate an SQL query based on file input and a natural language query."
    )
    parser.add_argument("input_file", help="Path to the file containing schema or relevant context.")
    parser.add_argument("query", help="The question you want to generate an SQL query for.")
    args = parser.parse_args()

    # Read file content (e.g., table schema)
    try:
        with open(args.input_file, "r") as f:
            file_content = f.read()
    except Exception as e:
        print(f"Error reading file '{args.input_file}': {e}")
        return

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Build the system prompt using the file content as context.
    system = (
        "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        "Consider the table defined in the schema provided below:\n\n"
        f"{file_content}\n\n"
        "Write a valid SQLite SQL query that answers the following question. "
        "Make sure the query ends with a semicolon."
    )

    # Build the user prompt with the question.
    user = f"Question: {args.query}\n"

    # Combine system and user into one prompt string.
    prompt = system + "\n" + user
    logger.info("Created prompt for Lamini LLM.")

    # Create a Lamini LLM instance.
    llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # Generate the SQL query using the LLM.
    result = llm.generate(prompt, output_type={"sql_query": "str"}, max_new_tokens=200)

    # Print the generated SQL query.
    print("Generated SQL Query:")
    print(result.get("sql_query", "No query was generated."))

if __name__ == "__main__":
    main()
