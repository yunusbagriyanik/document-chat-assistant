from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

"""
Build a conversational retrieval prompt template.

Returns:
    ChatPromptTemplate: Constructed prompt template.
"""


def generate_chat_prompt() -> ChatPromptTemplate:
    system_template = """
         If you cannot find the answer from the pieces of context, just say Sorry I Can't Help You.
         ----------------
         {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    return ChatPromptTemplate.from_messages(messages)
