import gradio as gr
import transformers
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from datasets import Dataset, load_dataset, DatasetDict

import requests
import torch
import json
from datetime import datetime
import operator

from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported # UNSLOTH installation via PIP: https://github.com/unslothai/unsloth/blob/main/README.md
PatchDPOTrainer()
from transformers import TrainingArguments, TextStreamer
from trl import ORPOTrainer, ORPOConfig, DPOTrainer, KTOTrainer, KTOConfig

# ================================= POSTGRESQL =================================

record_id = 0
rlhf_type = "ORPO"

def connect_postgresql():
    import psycopg2
    try:
        connection = psycopg2.connect(
            user="user",
            password="pass",
            host="172.16.59.1",
            port="5454",
            database="gradio"
        )
        cursor = connection.cursor()
        return connection, cursor
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
        
def close_postgresql(connection, cursor):
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")

def create_completion_table_postgresql():
    connection, cursor = connect_postgresql()
    create_table_query = '''CREATE TABLE IF NOT EXISTS completion_generator
          (ID SERIAL PRIMARY KEY NOT NULL,
          SYSTEM TEXT NOT NULL,
          PROMPT TEXT NOT NULL,
          ANSWER TEXT NOT NULL,
          FEEDBACK FLOAT8); '''
    cursor.execute(create_table_query)
    connection.commit()
    close_postgresql(connection, cursor)
    
def create_dpo_table_postgresql():
    # columns system, prompt, chosen, rejected
    connection, cursor = connect_postgresql()
    create_table_query = '''CREATE TABLE IF NOT EXISTS dpo_dataset
            (ID SERIAL PRIMARY KEY NOT NULL,
            SYSTEM TEXT NOT NULL,
            QUESTION TEXT NOT NULL,
            CHOSEN TEXT NOT NULL,
            REJECTED TEXT NOT NULL); '''
    cursor.execute(create_table_query)
    connection.commit()
    close_postgresql(connection, cursor)
    
def create_kto_table_postgresql():
    # columns system, prompt, completion, label (true, false), rating (feedback)
    connection, cursor = connect_postgresql()
    create_table_query = '''CREATE TABLE IF NOT EXISTS kto_dataset
            (ID SERIAL PRIMARY KEY NOT NULL,
            SYSTEM TEXT NOT NULL,
            PROMPT TEXT NOT NULL,
            COMPLETION TEXT NOT NULL,
            LABEL BOOLEAN NOT NULL,
            RATING FLOAT8); '''
    cursor.execute(create_table_query)
    connection.commit()
    close_postgresql(connection, cursor)
    
def create_orpo_table_postgresql():
    # columns instruction, input, accepted, rejected
    connection, cursor = connect_postgresql()
    create_table_query = '''CREATE TABLE IF NOT EXISTS orpo_dataset
            (ID SERIAL PRIMARY KEY NOT NULL,
            INSTRUCTION TEXT NOT NULL,
            INPUT TEXT NOT NULL,
            ACCEPTED TEXT NOT NULL,
            REJECTED TEXT NOT NULL); '''
    cursor.execute(create_table_query)
    connection.commit()
    close_postgresql(connection, cursor)
    
def from_completion_to_dpo():
    create_dpo_table_postgresql()
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT * from completion_generator")
    records = cursor.fetchall()
    # to dataframe
    df = pd.DataFrame(records, columns=["id", "system", "prompt", "answer", "feedback"])
    df['tup'] = list(zip(df['answer'], df['feedback']))
    #grouping together all the answers for a given question along with its feedback
    df_g = df.groupby('prompt')['tup'].apply(list).reset_index()
    print(df_g.dtypes)
    # sort each group based on the feedback score
    df_g["sorted_tup"] = df_g["tup"].apply(lambda x :sorted(x,key=operator.itemgetter(1)) )
    # answer with highest feedback score is "chosen"
    df_g["chosen"] = df_g["sorted_tup"].apply(lambda x: x[-1][0])
    df_g["chosen_score"] = df_g["sorted_tup"].apply(lambda x: x[-1][1])
    # answer with highest feedback score is "rejected"
    df_g["rejected"] = df_g["sorted_tup"].apply(lambda x: x[0][0])
    df_g["rejected_score"] = df_g["sorted_tup"].apply(lambda x: x[0][1])
    df_g = df_g.dropna()
    # mantain only when chosen_score >= 4.0 and rejected_score < 4.0
    df_g = df_g[(df_g["chosen_score"] >= 4.0) & (df_g["rejected_score"] < 4.0)]

    rows = []
    for record in df_g.itertuples(index=True, name='Pandas'):
        if record is None or len(record) == 0:
            continue
        rows.append({
            "prompt": record.prompt,
            "chosen": record.chosen,
            "rejected": record.rejected
        })
        
    system_text = "You are an AI requirements finder. Generate a list with all the requirements found in the text below:"
    
    for row in rows:
        print(row)
        insert_query = """ INSERT INTO dpo_dataset (SYSTEM, QUESTION, CHOSEN, REJECTED) VALUES (%s, %s,%s,%s)"""
        record_to_insert = (system_text, row["prompt"], row["chosen"], row["rejected"])
        cursor.execute(insert_query, record_to_insert)
        
    connection.commit()
    close_postgresql(connection, cursor)
    
def from_completion_to_kto():
    create_kto_table_postgresql()
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT * from completion_generator")
    records = cursor.fetchall()
    # to dataframe
    df = pd.DataFrame(records, columns=["id", "system", "prompt", "answer", "feedback"])
    # loop through the answers
    df['label'] = np.where(df['answer'].str.contains("JSON"), 'True', 'False')
    df = df.rename(columns={"answer": "completion"})
    df = df.rename(columns={"feedback": "rating"})
    
    # insert in the kto_dataset table
    for row in df.itertuples(index=True, name='Pandas'):
        insert_query = """ INSERT INTO kto_dataset (SYSTEM, PROMPT, COMPLETION, LABEL, RATING) VALUES (%s,%s,%s,%s,%s)"""
        record_to_insert = (row.system, row.prompt, row.completion, row.label, row.rating)
        cursor.execute(insert_query, record_to_insert)
        
    connection.commit()
    close_postgresql(connection, cursor)

def from_dpo_to_orpo():
    create_orpo_table_postgresql()
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT * from dpo_dataset")
    records = cursor.fetchall()
    # to dataframe
    df = pd.DataFrame(records, columns=["id", "system", "question", "chosen", "rejected"])
    df = df.rename(columns={"system": "instruction"}) # rename prompt to question
    df = df.rename(columns={"question": "input"}) # rename prompt to question
    df = df.rename(columns={"chosen": "accepted"}) # rename prompt to question
    
    # insert in the kto_dataset table
    for row in df.itertuples(index=True, name='Pandas'):
        insert_query = """ INSERT INTO orpo_dataset (INSTRUCTION, INPUT, ACCEPTED, REJECTED) VALUES (%s,%s,%s,%s)"""
        record_to_insert = (row.instruction, row.input, row.accepted, row.rejected)
        cursor.execute(insert_query, record_to_insert)
        
    connection.commit()
    close_postgresql(connection, cursor)
    
def drop_completion_table_postgresql():
    connection, cursor = connect_postgresql()
    drop_table_query = '''DROP TABLE completion_generator'''
    cursor.execute(drop_table_query)
    connection.commit()
    close_postgresql(connection, cursor)
    
def insert_record_postgresql(system, prompt, answer, feedback):
    connection, cursor = connect_postgresql()
    postgres_insert_query = """ INSERT INTO completion_generator (SYSTEM, PROMPT, ANSWER, FEEDBACK) VALUES (%s,%s,%s,%s)"""
    record_to_insert = (system, prompt, answer, feedback)
    cursor.execute(postgres_insert_query, record_to_insert)
    connection.commit()
    close_postgresql(connection, cursor)
    
def select_records_postgresql():
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT * from completion_generator")
    records = cursor.fetchall()
    close_postgresql(connection, cursor)
    return records

def select_records_customdb_postgresql(db):
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT * from {}".format(db))
    records = cursor.fetchall()
    close_postgresql(connection, cursor)
    return records

def select_record_postgresql(key):
    global record_id
    if key == 0:
        record_id = 1
    elif key == 1:
        if record_id > 1:
            record_id = record_id - 1
    elif key == 2:
        record_id = record_id + 1
    connection, cursor = connect_postgresql()
    select_query = """SELECT * from completion_generator WHERE ID = %s"""
    cursor.execute(select_query, (record_id,))
    records = cursor.fetchone()
    close_postgresql(connection, cursor)
    if records is None:
        print("No records found.")
        return "No records found.", "No records found.", "No records found.", 0
    if records[4] is None:
        records[4] = 0 
    return records[1], records[2], records[3], records[4]

def select_single_record_postgresql(key, column):
    global record_id
    if key == 0:
        record_id = 1
    elif key == 1:
        record_id = record_id - 1
    elif key == 2:
        record_id = record_id + 1
    
    connection, cursor = connect_postgresql()
    select_query = """SELECT {} FROM completion_generator WHERE ID = %s""".format(column)
    cursor.execute(select_query, (str(record_id)))
    record = cursor.fetchone()
    close_postgresql(connection, cursor)
    # if record is empty, return a message
    if record is None:
        print("No records found.")
        return "No records found."
    else:
        for row in record:
            row = row

def select_single_record_and_feedback_postgresql(key, column):
    global record_id
    if key == 0:
        record_id = 1
    elif key == 1:
        record_id = record_id - 1
    elif key == 2:
        record_id = record_id + 1
    
    connection, cursor = connect_postgresql()
    select_query = """SELECT PROMPT, {}, FEEDBACK FROM completion_generator WHERE ID = %s""".format(column)
    cursor.execute(select_query, (str(record_id)))
    record = cursor.fetchone()
    close_postgresql(connection, cursor)
    if record[0] is None:
        record[0] = "No records found." 
    if record[1] is None:
        record[1] = "No records found." 
    if record[2] is None:
        record[2] = 0 
    
    return record[0], record[1], record[2]

def delete_record_postgresql(id):
    connection, cursor = connect_postgresql()
    delete_query = """DELETE FROM completion_generator WHERE ID = %s"""
    cursor.execute(delete_query, (id,))
    connection.commit()
    close_postgresql(connection, cursor)
    
def update_record_postgresql(id, system, prompt, answer, feedback):
    connection, cursor = connect_postgresql()
    update_query = """UPDATE completion_generator SET SYSTEM = %s, PROMPT = %s, ANSWER = %s, FEEDBACK = %s WHERE ID = %s"""
    cursor.execute(update_query, (system, prompt, answer, feedback, id))
    connection.commit()
    close_postgresql(connection, cursor)
    
def update_feedback_postgresql(feedback):
    global record_id
    connection, cursor = connect_postgresql()
    update_query = """UPDATE completion_generator SET FEEDBACK = %s WHERE ID = %s"""
    cursor.execute(update_query, (feedback, record_id))
    connection.commit()
    close_postgresql(connection, cursor)
    
def count_records_postgresql():
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT COUNT(*) from completion_generator")
    records = cursor.fetchone()
    close_postgresql(connection, cursor)
    return records[0]

def count_feedback_records_postgresql():
    connection, cursor = connect_postgresql()
    cursor.execute("SELECT COUNT(*) from completion_generator WHERE FEEDBACK > 0")
    records = cursor.fetchone()
    close_postgresql(connection, cursor)
    total_records = count_records_postgresql()
    label1 = gr.Label(f"Total Records: {total_records}")
    label2 = gr.Label(f"Feedback Records: {records[0]}")
    return label1, label2

# =======================================================================

# create an empty df
df = pd.DataFrame(columns=["prompt", "answer", "feedback"])

theme = gr.themes.Default(primary_hue="blue").set(
    button_secondary_background_fill="#04ec04",
)

def llm_finetuned(prompt, load_model=True, model_path="/notebooks/_meta-llama/unsloth/RequistosJSON/LORA-RLORPO-llama-3-8b-Instruct-bnb-4bit", 
                  max_seq_length=2048, load_in_4bit=True):
    global model_rlhf
    global tokenizer_rlhf
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        if (torch.cuda.memory_allocated(device) / 1e9) > 5:
            load_model = False
            print("Model not loaded.")
    
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction & Input:
    {}

    ### Response:
    {}"""
    
    if load_model:
        model_rlhf, tokenizer_rlhf = FastLanguageModel.from_pretrained(
            model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model_rlhf) # Enable native 2x faster inference
    else: FastLanguageModel.for_inference(model_rlhf) # Enable native 2x faster inference

    inputs = tokenizer_rlhf(
    [
        alpaca_prompt.format(
            # "You are an AI requirements finder. Generate a list with all the requirements found in the text below:", # instruction
            # "The system shall give the name of the current football player and pop up an alert.", # input
            prompt,
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model_rlhf.generate(**inputs, max_new_tokens = 256, use_cache = True)
    decoded_output = tokenizer_rlhf.decode(outputs[0], skip_special_tokens = True)
    return decoded_output

def llm_endpoint(text_input, endpoint_url = "http://172.16.59.1:8000/v2/models/ensemble/generate"):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "text_input": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{text_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
        "parameters": {
            "max_tokens": 2048,
            "bad_words": [""],
            "stop_words": [""]
        }
    }
    response = requests.post(endpoint_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["text_output"]
    else:
        response.raise_for_status()

def generate_completions(system, prompt, prompt_file):
    try:
        if prompt_file:
            data = pd.read_json(prompt_file)
            data = data[["system", "prompt"]].values
        else:
            data = [(system, prompt)]
    except Exception as e:
        print("Failed to read the file.")
        return
    

    answers = []

    for system, prompt in data:
        # concatenate system and prompt
        user_prompt = f"{system} {prompt}"
        # answer = inference(pipeline, system, prompt)
        answer = llm_endpoint(user_prompt)
        if answer:
            answers.append({"prompt": prompt, "answer": answer, "feedback": ""})
            insert_record_postgresql(system, prompt, answer, 0)
        else:
            print("Failed to generate an answer.")
    
    df = pd.DataFrame(answers)
    return df

def load_quantized_llama3(model_name, max_seq_length, load_in_4bit):
    global model
    global tokenizer
    # global rlhf_type
    # rlhf_type = rlhf_name
    
    # model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    # max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    # dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    # load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    
    return "Model loaded."

def load_peft_llama3(PEFT_rank, PEFT_lora_alpha, PEFT_lora_dropout):
    global model
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank. The rank of the matrix decomposition. Higher rank = more precission. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],  # According to the QLoRA paper, the most important thing you can do to make LoRA fine-tuning effective is to train all layers of the network. Then, and only then, were they able to achieve the quality of full-parameter fine-tuning.
        lora_alpha = 16, # Decreasing alpha relative to rank increases the effect of fine-tuning. Increasing alpha relative to rank decreases it. If you set alpha to the same as rank, then you’re adding the weight changes as they were determined during training (multiplied by 1x), which makes the most sense to me.
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return "PEFT model loaded."

def load_dataset(db):
    # to df
    if db == "DPO":
        records = select_records_customdb_postgresql("dpo_dataset")
        df = pd.DataFrame(records, columns=["id", "system", "question", "chosen", "rejected"])
    elif db == "KTO":
        records = select_records_customdb_postgresql("kto_dataset")
        df = pd.DataFrame(records, columns=["id", "system", "prompt", "completion", "label", "rating"])
    elif db == "ORPO":
        records = select_records_customdb_postgresql("orpo_dataset")
        df = pd.DataFrame(records, columns=["id", "instruction", "input", "accepted", "rejected"])
    
    train_df = df.iloc[:int(len(df)*0.8)]
    val_df = df.iloc[int(len(df)*0.8):]
    
    print(f"Training dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    tds = Dataset.from_pandas(train_df)
    vds = Dataset.from_pandas(val_df)
    ds = DatasetDict()
    ds['train'] = tds
    ds['validation'] = vds
    print(ds)
    return ds
    
def chatml_format(example):
    global tokenizer
    global rlhf_type
    EOS_TOKEN = tokenizer.eos_token

    if rlhf_type == "DPO":
        dpo_args = {
            "system": example['system'],
            "prompt": example['question'],
            "chosen": example['chosen'],
            "rejected": example['rejected'],
        }
        kto_args = {}
        orpo_args = {}
    elif rlhf_type == "KTO":    
        dpo_args = {}
        kto_args = {
            "system": example['system'],
            "prompt": example['prompt'],
            "completion": example['completion'],
            "label": example['label'],
        }
        orpo_args = {}
    elif rlhf_type == "ORPO":
        dpo_args = {}    
        kto_args = {}
        orpo_args = {
            "system": example['instruction'],
            "prompt": example['input'],
            "chosen": example['accepted'],
            "rejected": example['rejected'],
        }
    
    rlhf_args = {
        "DPO": dpo_args,
        "KTO": kto_args,
        "ORPO": orpo_args,
    }
    
    # Format system
    if len(rlhf_args[rlhf_type]["system"]) > 0:
        message = {"role": "system", "content": rlhf_args[rlhf_type]['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": rlhf_args[rlhf_type]['prompt']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    if rlhf_type == "DPO" or rlhf_type == "ORPO":
        # Format chosen answer
        chosen = rlhf_args[rlhf_type]['chosen'] + EOS_TOKEN
        # Format rejected answer
        rejected = rlhf_args[rlhf_type]['rejected'] + EOS_TOKEN
        return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
        }
    else:
        # Format completion
        completion = rlhf_args[rlhf_type]['completion'] + EOS_TOKEN
        # Format label
        label = rlhf_args[rlhf_type]['label']
        if label == "False":
            label = False
        else:
            label = True
        return {
            "prompt": system + prompt,
            "completion": completion,
            "label": label,
        }
    
def train(rlhf_name, max_seq_length, per_device_train_batch_size, per_device_eval_batch_size, gradient_accumulation_steps, max_steps, warmup_steps, learning_rate):
    global rlhf_type
    rlhf_type = rlhf_name
    
    # Load dataset
    dataset = load_dataset(rlhf_type)
    original_columns = dataset['train'].column_names
    
    # Format dataset
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns
    )
    
    # print all the function arguments  
    print(f"RLHF Type: {rlhf_type}\n\n Max Seq Length: {max_seq_length}\n\n Per Device Train Batch Size: {per_device_train_batch_size}\n\n Per Device Eval Batch Size: {per_device_eval_batch_size}\n\n Gradient Accumulation Steps: {gradient_accumulation_steps}\n\n Max Steps: {max_steps}\n\n Warmup Steps: {warmup_steps}\n\n Learning Rate: {learning_rate}")
    
    PatchDPOTrainer()
    if rlhf_type == "DPO":
        trainer = DPOTrainer(
            model,
            args=TrainingArguments(
                per_device_train_batch_size = per_device_train_batch_size,
                gradient_accumulation_steps = gradient_accumulation_steps,
                warmup_steps = warmup_steps,
                max_steps = max_steps,
                learning_rate = learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                report_to="none",
                output_dir = "outputs/",
            ),
            beta=0.1,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'], 
            tokenizer=tokenizer,
            max_length=1024,
            max_prompt_length=512,
        )
    elif rlhf_type == "KTO":
        KTOTrainer(
            model,
            args=KTOConfig(
                beta=0.2,
                desirable_weight=1.33,
                undesirable_weight=1,
                eval_strategy= "steps",
                per_device_train_batch_size= per_device_train_batch_size,
                per_device_eval_batch_size= per_device_eval_batch_size,
                gradient_accumulation_steps= gradient_accumulation_steps,
                eval_steps= 4,
                save_steps= 4,
                logging_steps= 1,
                learning_rate= learning_rate,
                num_train_epochs= 1,
                lr_scheduler_type= "cosine",
                warmup_steps= warmup_steps,
                bf16= is_bfloat16_supported(),
                fp16= not is_bfloat16_supported(),
                optim= "paged_adamw_8bit",
                load_best_model_at_end= True,
                save_total_limit= 1,
                weight_decay = 0.01,
                report_to="none",
                output_dir= "outputs/"
            ),
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'], 
            tokenizer=tokenizer,
        )
    elif rlhf_type == "ORPO":
        trainer = ORPOTrainer(
            model,
            args=ORPOConfig(
                eval_strategy= "steps",
                max_length = max_seq_length,
                max_prompt_length = max_seq_length//2,
                max_completion_length = max_seq_length//2,
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size = per_device_eval_batch_size,
                eval_steps = 1,
                save_steps= 4,
                gradient_accumulation_steps = gradient_accumulation_steps,
                beta = 0.1,
                logging_steps = 1,
                optim = "adamw_8bit",
                learning_rate= learning_rate,
                lr_scheduler_type = "cosine",
                num_train_epochs = 1,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                load_best_model_at_end= True,
                save_total_limit= 1,
                report_to="none",
                output_dir = "outputs/",
            ),
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'], 
            tokenizer=tokenizer,
        )
    
    trainer_stats = trainer.train()
    
    return trainer_stats
    
def save_model():
    model.save_pretrained_merged("LORA-RLORPO-llama-3-8b-Instruct-bnb-4bit", tokenizer, save_method = "lora",)
    return "Model saved."
    
def inference(user_prompt, chat_history, option):
    if option == "LLaMa 3":
        answer = llm_endpoint(user_prompt)
    else:
        answer = llm_finetuned(user_prompt)
    chat_history.append([user_prompt, answer])
    return " ", chat_history
 
with gr.Blocks(theme=theme) as demo:
    gr.HTML('<div style="display: flex; justify-content: start; background-color: rgb(37, 37, 37); width: 100vw; padding: 2rem; margin-bottom: 20px;"><img src="https://ikerlan-esports.eus/ikerlan/Ikerlan_logo_invertido.webp" alt="Ikerlan logo" height="24" width="100"/></div>')

    with gr.Tab("Rollout"):       
        with gr.Accordion("Prompts de ejemplo:", open=False):
                gr.Markdown("""
                    - En este paso... """)    
        system = gr.Textbox(placeholder="You are an AI document writer expert in...", label="Role: System")
        prompt = gr.Textbox(placeholder="Write the prompt.", label="Prompt")
        prompt_file = gr.File(label="JSON File")
        answers_df = gr.Dataframe(interactive=False)
        iface = gr.Interface(
            fn=generate_completions,
            inputs=[system, prompt, prompt_file],
            outputs=answers_df,
            title="Completion Generator",
            theme=theme,
            allow_flagging="manual",
            flagging_dir="flagged_data",
        )
        
    with gr.Tab("Evaluation"):
        
        with gr.Row():
            gr.Markdown("# Completion Evaluator")
        with gr.Row():
            with gr.Column():
                total_records = gr.Label("Total Records: 0")
                feedback_records = gr.Label("Feedback Records: 0")
                btn_refresh = gr.Button("🔄")
            with gr.Row():
                btn_DPO = gr.Button("Generate DPO dataset")
                btn_KTO = gr.Button("Generate KTO dataset")
                btn_ORPO = gr.Button("Generate ORPO dataset")
        with gr.Row():
            btn_left = gr.Button("⬅️")
            btn_right = gr.Button("➡️")
        with gr.Row():
            feedback = gr.Slider(minimum=0, maximum=5, step=0.5, label="Feedback")
        with gr.Row():
            system = gr.Textbox(placeholder="System", label="System")
            prompt = gr.Textbox(placeholder="Prompt", label="Prompt")
        completions = gr.Markdown("Completion")
        
        # Listeners
        btn_DPO.click(from_completion_to_dpo)
        btn_KTO.click(from_completion_to_kto)
        btn_ORPO.click(from_dpo_to_orpo)
        btn_left.click(select_record_postgresql, inputs=[gr.Number(1, visible=False)], outputs=[system, prompt, completions, feedback])
        btn_right.click(select_record_postgresql, inputs=[gr.Number(2, visible=False)], outputs=[system, prompt, completions, feedback])
        btn_refresh.click(count_feedback_records_postgresql, outputs=[total_records, feedback_records])
        feedback.change(update_feedback_postgresql, inputs=[feedback])
        
    with gr.Tab("Fine-tuning"):
        with gr.Row():
            gr.Markdown("# RLHF Fine-Tuning")
        
        model_name = gr.Dropdown(choices=["unsloth/llama-3-8b-Instruct-bnb-4bit",], label="Model", type="value", multiselect=False, value="unsloth/llama-3-8b-Instruct-bnb-4bit")
        max_seq_length = gr.Slider(minimum=2, maximum=2048, step=2, label="Max Seq Length", value=2048)
        load_in_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
                
        iface = gr.Interface(
            fn=load_quantized_llama3,
            inputs=[model_name, max_seq_length, load_in_4bit],
            outputs=gr.Label("Load the base model."),
            title="Load Quantized Llama-3",
            theme=theme,
            allow_flagging="never",
        )
        
        PEFT_rank = gr.Slider(minimum=0, maximum=128, step=1, label="PEFT Rank (Suggested 8,16,32,64,128)", value=16)
        PEFT_lora_alpha = gr.Slider(minimum=0, maximum=256, step=1, label="PEFT LoRA Alpha", value=16)
        PEFT_lora_dropout = gr.Slider(minimum=0, maximum=1, step=0.01, label="PEFT LoRA Dropout", value=0)
                
        iface2 = gr.Interface(
            fn=load_peft_llama3,
            inputs=[PEFT_rank, PEFT_lora_alpha, PEFT_lora_dropout],
            outputs=gr.Label("Load the PEFT model."),
            title="QLoRA",
            theme=theme,
            allow_flagging="never",
        )
        
        rlhf_type = gr.Dropdown(choices=["DPO", "KTO", "ORPO"], label="RLHF Type", type="value", multiselect=False, value="ORPO")
        per_device_train_batch_size = gr.Slider(minimum=1, maximum=16, step=1, label="Per Device Train Batch Size", value=2)
        per_device_eval_batch_size = gr.Slider(minimum=1, maximum=16, step=1, label="Per Device Eval Batch Size", value=2)
        gradient_accumulation_steps = gr.Slider(minimum=1, maximum=16, step=1, label="Gradient Accumulation Steps", value=4)
        max_steps = gr.Slider(minimum=1, maximum=150, step=1, label="Max Steps", value=60)
        warmup_steps = gr.Slider(minimum=1, maximum=100, step=1, label="Warmup Steps", value=5)
        learning_rate = gr.Number(label="Learning Rate", value=2e-4)
        
        iface3 = gr.Interface(
            fn=train,
            inputs=[rlhf_type, max_seq_length, per_device_train_batch_size, per_device_eval_batch_size, 
                    gradient_accumulation_steps, max_steps, warmup_steps, learning_rate],
            outputs=gr.JSON(),
            title="Training",
            theme=theme,
            allow_flagging="never",
        )
        
        iface4 = gr.Interface(
            fn=save_model,
            inputs=[],
            outputs=gr.Markdown(),
            title="Save the weights.",
            theme=theme,
            allow_flagging="never",
        )
        
    with gr.Tab("Inference"):
        option = gr.Radio(choices=["LLaMa 3", "RLHF LLaMa3"], value="LLaMa 3")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Write a message...", label="Message")
        clear = gr.ClearButton([msg, chatbot])
        
        msg.submit(inference, inputs=[msg, chatbot, option], outputs=[msg, chatbot])
        
if __name__ == "__main__":
    # drop_completion_table_postgresql()
    create_completion_table_postgresql()
    # load_quantized_llama3()
    demo.launch()