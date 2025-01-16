from transformers import XLNetForSequenceClassification, XLNetTokenizer
import torch
import mlflow
import accelerate


class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model snapshot directory.
        """
        model_name = "xlnet-base-cased"
        self.model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

        # NB: If you do not have a CUDA-capable device or have torch installed with CUDA support
        # this setting will not function correctly. Setting device to 'cpu' is valid, but
        # the performance will be very slow.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=device)
        # If running on a GPU-compatible environment, uncomment the following line:
        # self.model.to(device="cuda")

        self.model.eval()

    def _build_prompt(self, question, max_len=32):
        # """
        # This method generates the prompt for the model.
        # """
        # INSTRUCTION_KEY = "### Instruction:"
        # RESPONSE_KEY = "### Response:"
        # INTRO_BLURB = (
        #     "Below is an instruction that describes a task. "
        #     "Write a response that appropriately completes the request."
        # )

        # return f"""{INTRO_BLURB}
        # {INSTRUCTION_KEY}
        # {instruction}
        # {RESPONSE_KEY}
        # """
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}

    def predict(self, question, max_len=32, device="cpu"):
        """
        This method generates prediction for the given input.
        """
        # Build the prompt
        preprocessed_question = self._build_prompt(question, max_len)

        input_ids = preprocessed_question["input_ids"].to(device)
        attention_mask = preprocessed_question["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        return {"predicted_class": predicted_class, "probabilities": probabilities.tolist()}

    def infer(self, question):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = self.predict(question, max_len=32, device=device)
        label_map = {0: "general", 1: "concreta"}
        predicted_label = label_map[output["predicted_class"]]

        return f'La pregunta "{question}" es clasificada como: {predicted_label}'
