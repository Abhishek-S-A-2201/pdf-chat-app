import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMResponse(BaseModel):
    answer: str
    citations: List[str]

class QASystem:
    """
    A Question-Answering system that uses a language model to answer queries
    based on provided context and chat history.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.5):
        """
        Initializes the QA system.

        Args:
            model_name: The name of the OpenAI model to use.
            temperature: The creativity of the model's responses (0.0 to 1.0).
        """
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.model = self.model.with_structured_output(LLMResponse)
        self.chat_history: List[HumanMessage | AIMessage] = []

        # The prompt template is the heart of the RAG system.
        # It instructs the model on how to behave.
        self.prompt = self._create_prompt_template()
        
        # Build the RAG chain using LangChain Expression Language (LCEL)
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=self._format_context  # Format the retrieved chunks
            )
            | self.prompt
            | self.model
        )
        logging.info("QA System initialized successfully.")

    @staticmethod
    def _create_prompt_template() -> ChatPromptTemplate:
        """Creates the prompt template for the RAG chain."""
        template = """
        You are a specialized AI assistant with an expertise in analyzing and summarizing information. Your primary goal is to answer questions accurately based on the provided document 'Context' and the 'Chat History' of our conversation.

        **OUTPUT FORMAT:**
        Your final output must be a single, valid JSON object that adheres to the following structure. Do not include any text, explanations, or markdown formatting outside of this JSON object.
        ```json
        {{
          "answer": "A concise, direct answer to the user's question.",
          "citations": [
            "A direct quote from the 'Context' that supports the answer.",
            "Another supporting quote from the 'Context'.",
            "A summary of information taken from the 'Chat History'."
          ]
        }}
        ```

        **INSTRUCTIONS:**
            1.  **Analyze All Information:** Carefully read the user's 'Question', the provided 'Context' from the document, and our 'Chat History'.
            2.  **Synthesize Your Answer:** Formulate a concise and direct answer based on the most relevant information available across all sources.
            3.  **Prioritize and Ground:**
                * Your answer MUST be grounded in the 'Context' or the 'Chat History'. Do not use any external knowledge.
                * **Prioritize the 'Context' as the primary source of truth.** If the 'Context' contains relevant information, use it.
                * Use the 'Chat History' to answer questions that refer to things we've already discussed, especially if the new 'Context' doesn't cover it.
            4.  **Provide References:** After your answer, add a "References" section.
                * If the information came from the document, quote the exact sentence(s) from the 'Context'.
                * If the information came from our conversation, state it clearly. For example: "Reference: From our previous conversation, we established that..."
            5.  **Handle Conflicts and Unknowns:**
                * If you are not 100 percent sure of the answer, you MUST respond with: "I don't know.".
                * If the 'Context' and 'Chat History' have conflicting information, prioritize the 'Context' and mention the discrepancy.
                * If NEITHER the 'Context' NOR the 'Chat History' contains the information needed, you MUST respond with: "I could not find the answer in the provided document or our conversation history."
            6.  **Maintain Context:** Use the 'Chat History' to understand follow-up questions (e.g., "what about the second one?") and to recall previously established facts.

            **Chat History:**
            {chat_history}

            **Context:**
            {context}

            **Question:**
            {question}

            **Answer:**
        """
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def _format_context(context_chunks: List[Dict[str, Any]]) -> str:
        """Formats the retrieved document chunks into a single string."""
        chunks = context_chunks.get("context_chunks", [])
        if len(chunks) == 0:
            return "No context provided."
        return "\n\n---\n\n".join([chunk["text"] for chunk in chunks])

    def answer_query(self, question: str, context_chunks: List[Dict[str, Any]]) -> LLMResponse:
        """
        Answers a query using the RAG chain and updates the chat history.

        Args:
            question: The user's query.
            context_chunks: The relevant document chunks from the vector store.

        Returns:
            The model's generated answer.
        """
        logging.info(f"Answering question: '{question}'")
        
        # Invoke the chain with all necessary inputs
        answer: LLMResponse = self.rag_chain.invoke({
            "question": question,
            "context_chunks": context_chunks,
            "chat_history": self.chat_history
        })
        
        # Update the chat history with the latest interaction
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer.answer)
        ])
        
        return answer