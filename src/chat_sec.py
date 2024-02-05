from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


def get_openai_answer_sec(question, context):
    sec_prompt_template = """Use the following pieces of context to answer the question at the end.\n
        Be very diligent in using all the information and answering extensively. Also, don't miss out any numerical figures if there are any.\n

        {context}

        Question: {question}
        """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"], template=sec_prompt_template
    )

    llm_prompt = prompt_template.format(question=question, context=context)

    earnings_call_llm = ChatOpenAI(
        temperature=0.0, model="gpt-3.5-turbo-16k", streaming=False
    )

    output = earnings_call_llm.predict(llm_prompt)
    return output
