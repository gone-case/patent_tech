"""
                 START
                   |
               harvester
                   |
                 intake ----(review=True)----> review (interrupt)
                   |                              |
                   +----------> planner <---------+
                                  |    ^
                               retrieve |
                                  |     | weak features (max MAX_ROUNDS)
                                score   |
                                  |     |
                                critic -+
                                  |
                      Send x top_k (parallel)
                                  |
                             claim_mapper
                                  |
                               reporter
                                  |
                                 END
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (claim_mapper_node, critic_node, harvester_node, intake_node, planner_node, reporter_node,
                    retrieve_node, review_node, route_after_critic, route_after_intake, score_node)
from .state import AgentState


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("harvester", harvester_node)
    g.add_node("intake", intake_node)
    g.add_node("review", review_node)
    g.add_node("planner", planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("score", score_node)
    g.add_node("critic", critic_node)
    g.add_node("claim_mapper", claim_mapper_node)
    g.add_node("reporter", reporter_node)

    g.add_edge(START, "harvester")
    g.add_edge("harvester", "intake")
    g.add_conditional_edges("intake", route_after_intake, ["review", "planner"])
    g.add_edge("review", "planner")
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "score")
    g.add_edge("score", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "claim_mapper", "reporter"])
    g.add_edge("claim_mapper", "reporter")
    g.add_edge("reporter", END)

    # The checkpointer keeps each run's state, so a run paused for human review can be resumed
    return g.compile(checkpointer=checkpointer or InMemorySaver())
