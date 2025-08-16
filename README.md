There are two folders and two respective setups to follow: LLMs and MLs.

**LLMs**


**Step 1**: Generate a small 100 text sample with the LLM Sampler. Skip if you are using the complete dataset.

**Step 2**: We have four independent scripts, one for each trait, where you can input the text directory, output directory, the prompt, the model, and your API key. The output will be in JSON format, one file for each text.

**Step 3**: Now, use JSON script to get an output CSV file where column 1 will be the text name and column 2 will be the prediction (you can name it ypred). Manually create a ytrue column from the original dataset for the next step.

**Step 4**: Run the Analysis script. You will have an output with a confusion matrix, and the respective evaluation metrics.
