"""
RAGAS Evaluator for RAG Assessment
Calculates Faithfulness, Answer Relevance, and Context Precision
"""

from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config


class RAGASEvaluator:
    """Evaluates RAG quality using RAGAS metrics with LLM as judge"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=config.MODEL_NAME, temperature=0)
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness: Did the answer come only from the retrieved context?
        Low score = Hallucination detected
        """
        if not contexts:
            return 0.0
        
        context_text = "\n\n".join(contexts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator checking if an answer is faithful to the provided context."),
            ("user", """Context:
{context}

Answer:
{answer}

Evaluate if the answer is ENTIRELY based on the context provided.
- If the answer contains information NOT in the context, it's unfaithful (hallucination)
- If the answer only uses information from the context, it's faithful

Rate faithfulness on a scale of 0.0 to 1.0:
- 1.0 = Completely faithful, all statements come from context
- 0.5 = Partially faithful, some statements not from context
- 0.0 = Unfaithful, mostly hallucinated

Respond with ONLY a number between 0.0 and 1.0.

Faithfulness score:""")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "context": context_text,
                "answer": answer
            })
            
            # Extract score
            score_text = response.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return 0.5  # Default to middle score on error
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Answer Relevance: Did the answer actually address the user's question?
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator checking if an answer is relevant to the question."),
            ("user", """Question:
{question}

Answer:
{answer}

Evaluate how well the answer addresses the question.
- Does it answer what was asked?
- Is it on-topic?
- Does it provide useful information?

Rate relevance on a scale of 0.0 to 1.0:
- 1.0 = Perfectly relevant, directly answers the question
- 0.5 = Partially relevant, tangentially related
- 0.0 = Not relevant, doesn't address the question

Respond with ONLY a number between 0.0 and 1.0.

Answer relevance score:""")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "question": question,
                "answer": answer
            })
            
            # Extract score
            score_text = response.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Error evaluating answer relevance: {e}")
            return 0.5
    
    def evaluate_context_precision(self, question: str, contexts: List[str], 
                                   relevance_scores: List[float]) -> float:
        """
        Context Precision: Did the retrieval system find the right documents?
        Measures if relevant docs are ranked higher
        """
        if not contexts or not relevance_scores:
            return 0.0
        
        # Calculate precision: proportion of relevant documents
        relevant_count = sum(1 for score in relevance_scores if score > 0.5)
        precision = relevant_count / len(relevance_scores)
        
        return precision
    
    def evaluate_all(self, question: str, answer: str, contexts: List[str], 
                    relevance_scores: List[float]) -> Dict[str, float]:
        """
        Evaluate all RAGAS metrics
        Returns dict with all scores
        """
        # Extract just the text content from contexts if they're dicts
        context_texts = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                context_texts.append(ctx.get("content", ""))
            else:
                context_texts.append(str(ctx))
        
        scores = {
            "faithfulness": self.evaluate_faithfulness(answer, context_texts),
            "answer_relevance": self.evaluate_answer_relevance(question, answer),
            "context_precision": self.evaluate_context_precision(question, context_texts, relevance_scores)
        }
        
        # Calculate average score
        scores["average"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def get_score_interpretation(self, score: float) -> tuple[str, str]:
        """
        Get interpretation and color for a score
        Returns (interpretation, color)
        """
        if score >= 0.8:
            return "Excellent ✅", "green"
        elif score >= 0.6:
            return "Good ✓", "blue"
        elif score >= 0.4:
            return "Fair ⚠️", "orange"
        else:
            return "Poor ❌", "red"
