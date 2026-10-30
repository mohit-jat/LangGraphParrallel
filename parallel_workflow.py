from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


class State(TypedDict):
    player: str
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    average: float
    total_runs: int
    boundaries: str
    boundary_runs: int
    final_summary: str


def calculate_strike_rate(state: State):
    strike_rate = (state["runs"] / state["balls"]) * 100

    return {
        "strike_rate": round(strike_rate, 2)
    }


def calculate_average(state: State):
    average = state["runs"] / state["balls"]

    return {
        "average": round(average, 2)
    }


def calculate_total_runs(state: State):
    return {
        "total_runs": state["runs"]
    }


def calculate_boundaries(state: State):
    boundaries = (
        f"Fours: {state['fours']}, "
        f"Sixes: {state['sixes']}"
    )

    return {
        "boundaries": boundaries
    }


def calculate_boundary_runs(state: State):
    boundary_runs = (
        (state["fours"] * 4) +
        (state["sixes"] * 6)
    )

    return {
        "boundary_runs": boundary_runs
    }


def final_summary(state: State):
    prompt = f"""
Give a short and professional cricket batting summary.

Player: {state['player']}
Total Runs: {state['total_runs']}
Balls Faced: {state['balls']}
Strike Rate: {state['strike_rate']}
Average Runs Per Ball: {state['average']}
Fours: {state['fours']}
Sixes: {state['sixes']}
Runs From Boundaries: {state['boundary_runs']}

Mention all important statistics.
Keep the summary simple and concise.
"""

    response = llm.invoke(prompt)

    return {
        "final_summary": response.content
    }


graph = StateGraph(State)

graph.add_node("strike_rate", calculate_strike_rate)
graph.add_node("average", calculate_average)
graph.add_node("total_runs", calculate_total_runs)
graph.add_node("boundaries", calculate_boundaries)
graph.add_node("boundary_runs", calculate_boundary_runs)
graph.add_node("final_summary", final_summary)

graph.add_edge(START, "strike_rate")
graph.add_edge(START, "average")
graph.add_edge(START, "total_runs")
graph.add_edge(START, "boundaries")
graph.add_edge(START, "boundary_runs")

graph.add_edge("strike_rate", "final_summary")
graph.add_edge("average", "final_summary")
graph.add_edge("total_runs", "final_summary")
graph.add_edge("boundaries", "final_summary")
graph.add_edge("boundary_runs", "final_summary")

graph.add_edge("final_summary", END)

app = graph.compile()

result = app.invoke({
    "player": "Virat Kohli",
    "runs": 82,
    "balls": 53,
    "fours": 7,
    "sixes": 2,
    "strike_rate": 0,
    "average": 0,
    "total_runs": 0,
    "boundaries": "",
    "boundary_runs": 0,
    "final_summary": ""
})

print("\n========== STRIKE RATE ==========")
print(result["strike_rate"])

print("\n========== AVERAGE ==========")
print(result["average"])

print("\n========== TOTAL RUNS ==========")
print(result["total_runs"])

print("\n========== BOUNDARIES ==========")
print(result["boundaries"])

print("\n========== BOUNDARY RUNS ==========")
print(result["boundary_runs"])

print("\n========== FINAL SUMMARY ==========")
print(result["final_summary"])