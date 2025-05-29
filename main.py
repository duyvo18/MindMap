from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

import os
import re
import pickle
import json


def rebuild_neo4j():
    load_dotenv()
    
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    ##############################build KG 

    session.run("MATCH (n) DETACH DELETE n")# clean all

    # read triples
    df = pd.read_csv('./data/chatdoctor5k/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])

    # Add tqdm to show progress
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Building Neo4j KG"):
        head_name = row['head']
        tail_name = row['tail']
        relation_name = row['relation']

        query = (
            "MERGE (h:Entity { name: $head_name }) "
            "MERGE (t:Entity { name: $tail_name }) "
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"
        )
        session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

def chat_extract_keyword(input_text, chat):
    system_message_template = "You are a medical entity extraction assistant. Explicitly extract the symptoms and their respective locations from medical questions. Always specify the location for each symptom, whether if it is stated or only implied."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    
    few_shot_examples = [
        {
            "input": "Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?",
            "entities": "Pain in vagina, Dryness in vagina, Pain during intercourse"
        },
        {
            "input": "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do?",
            "entities": "Panic attacks, Frequent panic attacks, Sudden panic attacks"
        },
    ]
    few_shot_template = PromptTemplate(
        input_variables=["input", "entities"],
        template="<CLS>{input}<SEP>The extracted entities are {entities}<EOS>"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=few_shot_template,
        prefix="Below are some examples of medical questions and their extracted entities.",
        suffix="<CLS>{input}<SEP>The extracted entities are ",
        input_variables=["input"]
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=few_shot_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input=input_text)
    messages = chat_prompt_with_values.to_messages()

    response_of_KG = chat.invoke(messages)

    re1 = r'(.*?)<EOS>$'
    question_kg = re.findall(re1, response_of_KG)
    if question_kg:
        question_kg = question_kg[0].split(",")
        question_kg = list(map(lambda x: x.strip(), question_kg))
    
    return question_kg

def embed_neo4j_entities():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    result = session.run("MATCH (e:Entity) RETURN e.name AS name")
    entities = [record["name"] for record in result]
    print(f"Found {len(entities)} results.")

    embed = OllamaEmbeddings(
        model="mxbai-embed-large:335m-v1-fp16",
        base_url="127.0.0.1:11434",
    )
    
    entity_embeddings = {
        "entities": [],
        "embeddings": []
    }
    for entity in tqdm(entities, desc="Embedding entities"):
        embeddings = np.array(embed.embed_query(entity.replace("_"," ")))
        entity_embeddings["entities"].append(entity)
        entity_embeddings["embeddings"].append(embeddings)

    with open("./data/chatdoctor5k/ollama_entity_embeddings.pkl", "wb") as f:
        pickle.dump(entity_embeddings, f)
    print("Embeds saved.")

def entity_extraction():
    chat = OllamaLLM(
        model="gemma3:4b-it-qat",
        base_url="127.0.0.1:11434",
        num_ctx=2048,
        temperature=1,
        keep_alive=10,
    )

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('./data/chatdoctor5k/ollama_entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)

    # with open('./data/chatdoctor5k/keyword_embeddings.pkl','rb') as f2:
    #     keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
   
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        for line in f.readlines()[-10:]:
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n","")
            input = input.replace("<OOS>","<EOS>")
            input = input.replace(":","") + "<END>"
            input_text = re.findall(re3,input)
            
            if input_text == []:
                continue
            print('Question:\n',input_text[0])

            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
                 
            question_kg = re.findall(re1,input)
            if len(question_kg) == 0:
                question_kg = re.findall(re2,input)
                if len(question_kg) == 0:
                    print("<Warning> no entities found", input)
                    continue
            question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
            question_kg = question_kg.replace("\n","")
            question_kg = question_kg.split(", ")
            print("question_kg:\n",question_kg)
            
            ollama_kg = chat_extract_keyword(input_text[0], chat)
            print("ollama_kg:\n",ollama_kg)
            
            
            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            embed = OllamaEmbeddings(
                model="mxbai-embed-large:335m-v1-fp16",
                base_url="127.0.0.1:11434",
                keep_alive=10,
            )
            for kg_entity in question_kg:
                kg_entity_emb = np.array(embed.embed_query(kg_entity))

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ","_"))
            print('match_kg:\n',match_kg)
            
            ollama_match_kg = []
            for kg_entity in ollama_kg:
                kg_entity_emb = np.array(embed.embed_query(kg_entity))
                
                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in ollama_match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                ollama_match_kg.append(match_kg_i.replace(" ","_"))
            print('ollama_match_kg:\n',ollama_match_kg)


if __name__ == "__main__":
    # embed_neo4j_entities()
    entity_extraction()
    
    
    # chat = OllamaLLM(
    #     model="gemma3:4b-it-qat",
    #     base_url="127.0.0.1:11434",
    #     num_ctx=2048,
    #     temperature=1,
    #     keep_alive=0,
    # )
    
    # input_text = "Doctor, I have an open wound on my nose, and I am experiencing hot flashes, facial pain, and diminished hearing. What could be the problem?"
    
    # system_message_template = "You are a medical entity extraction assistant. Extract explicit symptoms and their locations from medical questions."
    # system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)

    # few_shot_template = PromptTemplate(
    #     input_variables=["input", "entities"],
    #     template="<CLS>{input}<SEP>The extracted entities are {entities}<EOS>"
    # )
    
    # few_shot_examples = [
    #     {
    #         "input": "Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?",
    #         "entities": "Vaginal pain, Vaginal dryness, Pain during intercourse"
    #     },
    #     {
    #         "input": "Doctor, I have been experiencing sudden and frequent panic attacks. I don't know what to do?",
    #         "entities": "Panic attacks, Frequent panic attacks, Sudden panic attacks"
    #     },
    # ]

    # few_shot_prompt = FewShotPromptTemplate(
    #     examples=few_shot_examples,
    #     example_prompt=few_shot_template,
    #     prefix="Below are some examples of medical questions and their extracted entities.",
    #     suffix="<CLS>{input}<SEP>The extracted entities are",
    #     input_variables=["input"]
    # )

    # human_message_prompt = HumanMessagePromptTemplate(prompt=few_shot_prompt)
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # chat_prompt_with_values = chat_prompt.format_prompt(input=input_text)
    # messages = chat_prompt_with_values.to_messages()
    # print("messages:\n", messages)

    # response_of_KG = chat.invoke(messages)
    # print("response_of_KG:\n", response_of_KG)

    # re1 = r'(.*?)<EOS>$'
    # question_kg = re.findall(re1, response_of_KG)
    # if question_kg:
    #     question_kg = question_kg[0].split(",")
    #     question_kg = list(map(lambda x: x.strip(), question_kg))
    # print("question_kg:\n", question_kg)
    
    pass