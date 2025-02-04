import json
from openai import OpenAI
from llm_literature import Toolbox, Entity
import voyageai

available_functions = {
    "author_collaboration": Toolbox.author_collaboration,
    "certain_author": Toolbox.certain_author,
    "certain_entity": Toolbox.certain_entity,
}

author_collaboration_tool = {
    'type': 'function',
    'function': {
        'name': 'author_collaboration',
        'description': 'Access the information about the collaboration network of authors in certain area.',
        'parameters': {
            'type': 'object',
            'required': ['field'],
            'properties': {
                'field': {'type': 'string', 'description': 'The field of the discussion topic. Notice that every key point should be included. You should use the words in users initial query as much as possible.'},
            },
        },
    },
}

certain_author_tool = {
    'type': 'function',
    'function': {
        'name': 'certain_author',
        'description': 'Access the information about one of the authors in a certain area regarding articles and collaborations.',
        'parameters': {
            'type': 'object',
            'required': ['field', 'name'],
            'properties': {
                'field': {'type': 'string', 'description': 'The field of the discussion topic. Notice that every key point should be included. You should use the words in users initial query as much as possible.'},
                'name': {'type': 'string', 'description': 'The name of the author user wants to query.'},
            },
        },
    },
}

certain_entity_tool = {
    'type': 'function',
    'function': {
        'name': 'certain_entity',
        'description': 'The information about certain entity or concept',
        'parameters': {
            'type': 'object',
            'required': ['entity_name'],
            'properties': {
                'entity_name': {'type': 'string', 'description': 'The name of the entity or concept according to the query of the user. You should use the words in users initial query.'},
            },
        },
    },
}

chat_client = OpenAI(api_key="ollama", base_url='http://localhost:11434/v1')
embedding_client = voyageai.Client(api_key="your api key") 
Entity.set_embedding_client(embedding_client)

# Initialize variables
stored_field = "medical_research_using_LLM"  # Store the field for subsequent conversations

messages = [{'role': 'user', 'content': 'What is machine learning'}]

response = chat_client.chat.completions.create(
    model="qwen2.5:7b",  # Or a newer model that supports function calling
    messages=messages,
    tools=[author_collaboration_tool, certain_author_tool, certain_entity_tool],
    tool_choice="auto"
)

response_message = response.choices[0].message

if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        print(function_name)
        function_to_call = available_functions.get(function_name)
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "author_collaboration":
            print(function_name)
            field = stored_field
            if not stored_field and field:
                stored_field = field
            tools = Toolbox(field=field)
            function_response = tools.author_collaboration()  # 修改此处，移除 field 参数

        elif function_name == "certain_author":
            field = stored_field
            if not stored_field and field:
                stored_field = field
            name = function_args.get("name")
            tools = Toolbox(field=field)
            function_response = tools.certain_author(author=name)  # 修改此处，移除 field 参数

        elif function_name == "certain_entity":
            # field = stored_field # 移除此行，因为 certain_entity 不需要 field 参数
            entity_name = function_args.get("entity_name")
            tools = Toolbox(field="medical_research_using_LLM") 
            
            function_response = tools.certain_entity(name=entity_name)

        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  

    second_response = chat_client.chat.completions.create(
        model="qwen2.5:7b",
        messages=messages,
    )
    print(second_response.choices[0].message.content)
else:
    print("no tool was used")