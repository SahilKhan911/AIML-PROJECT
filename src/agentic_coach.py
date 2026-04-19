import os
from typing import TypedDict, List
from fpdf import FPDF
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from duckduckgo_search import DDGS

class CoachState(TypedDict):
    student_record: str
    student_goal: str
    diagnosis: str
    search_queries: List[str]
    retrieved_resources: str
    final_plan: str

class AgenticStudyCoach:
    def __init__(self, api_key: str, provider: str = "Google Gemini"):
        self.provider = provider
        if provider == "Google Gemini":
            # Using standard gemini model instead of pro if preferred, but gemini-1.5-flash is robust
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        elif provider == "Groq":
            self.llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider {provider}")

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(CoachState)
        workflow.add_node("diagnose", self.node_diagnose)
        workflow.add_node("search_resources", self.node_search)
        workflow.add_node("generate_plan", self.node_generate)

        workflow.add_edge(START, "diagnose")
        workflow.add_edge("diagnose", "search_resources")
        workflow.add_edge("search_resources", "generate_plan")
        workflow.add_edge("generate_plan", END)

        return workflow.compile()

    def node_diagnose(self, state: CoachState):
        prompt = f"""You are an expert AI Study Coach.
Based on the student's record and their goal, diagnose their learning gaps. Provide a concise summary and exactly 2 distinct search queries to find educational resources (e.g. tutorials, study guides).

Output format:
DIAGNOSIS:
[your diagnosis]
QUERIES:
[query 1]
[query 2]

Student Record:
{state['student_record']}

Student Goal:
{state['student_goal']}
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            return {"diagnosis": f"Error invoking LLM: {str(e)}", "search_queries": []}
            
        try:
            diagnosis_part = response.split("QUERIES:")[0].replace("DIAGNOSIS:", "").strip()
            queries_part = response.split("QUERIES:")[1].strip().split("\\n")
            queries = [q.strip("- *") for q in queries_part if q.strip()]
        except Exception:
            diagnosis_part = response
            queries = [state['student_goal'] + " study guide", state['student_goal'] + " tutorial"]

        return {"diagnosis": diagnosis_part, "search_queries": queries[:2]}

    def node_search(self, state: CoachState):
        ddgs = DDGS()
        resources = []
        for q in state.get("search_queries", []):
            try:
                # Get max 2 results per query
                results = ddgs.text(q, max_results=2)
                for r in results:
                    resources.append(f"- {r.get('title', 'Link')}: {r.get('href', '')}\\n  {r.get('body', '')}")
            except Exception as e:
                resources.append(f"Error searching {q}: {e}")
        
        return {"retrieved_resources": "\\n".join(resources)}

    def node_generate(self, state: CoachState):
        prompt = f"""You are an Expert AI Study Coach.
Create a comprehensive, multi-step personalized study plan for this student.
Incorporate their diagnosis and explicitly include the retrieved web resources as recommended reading.

Student Goal: {state['student_goal']}
Diagnosis: {state['diagnosis']}

Retrieved Resources:
{state['retrieved_resources']}

Format the output nicely using Markdown. Do not include introductory filler. 
Must include:
1. Learning Diagnosis
2. Personalized Study Plan
3. Weekly / Milestone-based Goals
4. Recommended Learning Resources (URLs)
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            response = f"Failed to generate study plan. Exception: {str(e)}"
            
        return {"final_plan": response}

    def run(self, student_record: str, student_goal: str) -> str:
        final_state = self.graph.invoke({
            "student_record": student_record,
            "student_goal": student_goal
        })
        return final_state.get("final_plan", "No plan generated.")


def generate_pdf_report(markdown_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Helvetica", style="", fname="", uni=False)
    pdf.set_font("Helvetica", size=12)

    # Basic markdown strip/convert logic for fpdf2
    lines = markdown_text.split('\\n')
    for line in lines:
        clean_line = line.replace('**', '').replace('*', '-').replace('#', '')
        # Handle unicode characters that fpdf default font does not support by replacing
        clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, txt=clean_line)

    return pdf.output()
