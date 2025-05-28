from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

import os
import re
import csv
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

def prompt_extract_keyword(input_text, chat):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat(chat_prompt_with_values.to_messages()).content

    re1 = r'The extracted entities are (.*?)<END>'
    question_kg = re.findall(re1,response_of_KG)
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

    ollama = OllamaEmbeddings(model="nomic-embed-text")
    
    entity_embeddings = []
    for entity in tqdm(entities, desc="Embedding entities"):
        embeddings = ollama.embed_query(entity)
        entity_embeddings.append({"entity": entity, "embeddings": embeddings})

    with open("./data/chatdoctor5k/ollama_entity_embeddings.pkl", "wb") as f:
        pickle.dump(entity_embeddings, f)

def entity_extraction():
    # OPENAI_API_KEY = os.getenv("YOUR_OPENAI_KEY")
    # chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
   
    with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        for line in f.readlines()[:10]:
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
            # output_text = re.findall(re3,output)
            # print(output_text[0])

                 
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

            answer_kg = re.findall(re1,output)
            if len(answer_kg) == 0:
                answer_kg = re.findall(re2,output)
                if len(answer_kg) == 0:
                    print("<Warning> no entities found", output)
                    continue
            answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
            answer_kg = answer_kg.replace("\n","")
            answer_kg = answer_kg.split(", ")
            # print("answer_kg:\n", answer_kg)

            
            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
           

            for kg_entity in question_kg:
                
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ","_"))
            print('match_kg:\n',match_kg)


if __name__ == "__main__":
    entity_extraction()
    pass